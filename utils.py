import logging
import os
import random
from typing import NamedTuple, List, Dict, Optional

import numpy as np
import pydicom
import torch
import warnings

from sybil import Serie


def disable_warnings() -> None:
    """
    Disable warnings
    """
    warnings.filterwarnings("ignore")


def detect_device() -> torch.device | str:
    """
    Detect device for computation
    :return: torch.device | str
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_global_seed(seed: int = 1997) -> None:
    """
    Set global seed for reproducibility
    :param seed: int
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class PatientScan(NamedTuple):
    """
    NamedTuple representing a patient scan with its associated dicom files in NLST dataset

    Attributes
    ----------
    `patient_id`: str
        Patient ID
    `scan_id`: str
        Scan ID
    `series_id`: str
        Series ID
    `dicom_paths`: List[str]
        List of dicom file paths

    Methods
    -------
    `valid_dicom_paths(slice_thickness_threshold: float = 5) -> List[str]`
        Filter out invalid dicom paths based on slice thickness

    `serie() -> Serie`
        Create a Serie instance from valid dicom paths if any

    `from_path(series_root: str, dicom_files: List[str]) -> PatientScan`
        Create a PatientScan instance from series root and dicom files
    """
    patient_id: str
    scan_id: str
    series_id: str
    dicom_paths: List[str]

    def valid_dicom_paths(self, slice_thickness_threshold: float = 5) -> List[str]:
        paths = []
        for path in self.dicom_paths:
            try:
                dicom = pydicom.dcmread(path, stop_before_pixels=True)
                if dicom.SliceThickness > slice_thickness_threshold:
                    logging.warning(
                        f"For dicom {path} slice thickness of {dicom.SliceThickness} is greater than "
                        f"threshold of {slice_thickness_threshold}"
                    )
                    continue

                paths.append(path)
            except Exception as e:
                logging.warning(f"Unable to read dicom file {path}: {e}")

        return sorted(paths)

    def serie(self) -> Optional[Serie]:
        dicom_paths = self.valid_dicom_paths()
        return Serie(dicom_paths) if dicom_paths else None

    @staticmethod
    def from_path(series_root: str, dicom_files: List[str]) -> "PatientScan":
        patient_id, scan_id, series_id = series_root.split(os.sep)[-3:]
        dicom_paths = [os.path.join(series_root, file) for file in dicom_files]

        return PatientScan(patient_id, scan_id, series_id, dicom_paths)


class PatientScanPrediction(NamedTuple):
    """
    NamedTuple representing a patient scan prediction

    Attributes
    ----------
    `patient_scan`: PatientScan
        Patient scan
    `serie`: Serie
        Serie instance
    `scores`: List[List[float]]
        Prediction scores
    `attentions`: List[Dict[str, np.ndarray]]
        Attention maps

    Methods
    -------
    `to_dict() -> Dict[str, np.ndarray]`
        Convert PatientScanPrediction to dictionary format
    """
    scan: PatientScan
    serie: Serie
    scores: List[float]
    attentions: List[Dict[str, np.ndarray]]

    def to_dict(self) -> Dict[str, np.ndarray]:
        metadata_dict = {
            "Patient ID": self.scan.patient_id,
            "Scan ID": self.scan.scan_id,
            "Series ID": self.scan.series_id
        }

        scores_dict = {
            f"Score Y{i + 1}": score
            for i, score in enumerate(self.scores)
        }

        return {
            **metadata_dict,
            **scores_dict
        }
