
import argparse
import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple
import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_batch", help="path to reference batch npz file")
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    args = parser.parse_args()

    config = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    print("warming up TensorFlow...")
    # This will cause TF to print a bunch of verbose stuff now rather
    # than after the next print(), to help prevent confusion.
    evaluator.warmup()

    print("computing reference batch activations...")
    ref_acts = evaluator.read_activations(args.ref_batch)
    print("computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(args.sample_batch)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample_batch, sample_acts)

    print("Computing evaluations...")
    print("Inception Score:", evaluator.compute_inception_score(sample_acts[0]))
    print("FID:", sample_stats.frechet_distance(ref_stats))
    print("sFID:", sample_stats_spatial.frechet_distance(ref_stats_spatial))
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    print("Precision:", prec)
    print("Recall:", recall)


class InvalidFIDException(Exception):
    pass


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(
        self,
        session,
        batch_size=64,
        softmax_batch_size=512,
    ):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.manifold_estimator = ManifoldEstimator(session)
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features for downstream evals.
        :param batches: a iterator over NHWC numpy arrays in [0, 255].
        :return: a tuple of numpy arrays of shape [N x X], where X is a feature
                 dimension. The tuple is (pool_3, spatial).
        """
        preds = []
        spatial_preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(spatial_preds, axis=0),
        )

    def read_statistics(
        self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()):
            return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
                obj["mu_s"], obj["sigma_s"]
            )
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i : i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def compute_prec_recall(
        self, activations_ref: np.ndarray, activations_sample: np.ndarray
    ) -> Tuple[float, float]:
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(
            activations_ref, radii_1, activations_sample, radii_2
        )
        return (float(pr[0][0]), float(pr[1][0]))


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.
    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        session,
        row_batch_size=10000,