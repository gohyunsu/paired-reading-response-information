from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from wordfreq import word_frequency


DATASETS = ("OneStop_RC", "SBSAT_RC")
REQUIRED_METADATA = {
    "unique_trial_id",
    "participant_id",
    "unique_paragraph_id",
    "question_id",
    "item_key",
    "label",
}


def _validate_metadata(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    missing = REQUIRED_METADATA - set(frame)
    if missing:
        raise ValueError(f"{dataset} metadata is missing columns: {sorted(missing)}")
    if frame.unique_trial_id.duplicated().any():
        raise ValueError(f"{dataset} trial identifiers are not unique")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError(f"{dataset} labels must contain both binary outcomes")
    out = frame.copy().reset_index(drop=True)
    out["scan_group"] = (
        out.participant_id.astype(str) + "::" + out.unique_paragraph_id.astype(str)
    )
    return out


def material_ids(metadata: pd.DataFrame, dataset: str) -> pd.Series:
    if dataset == "SBSAT_RC":
        return metadata.unique_paragraph_id.astype(str).copy()
    if dataset != "OneStop_RC":
        raise ValueError(f"Unknown dataset: {dataset}")
    parsed = metadata.unique_paragraph_id.astype(str).str.extract(
        r"^(\d+)_(\d+)_(Adv|Ele)_(\d+)$"
    )
    if parsed.isna().any().any():
        raise ValueError("Unrecognized OneStop paragraph identifier")
    return parsed[0] + "_" + parsed[1]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return 0.0
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def sequence_descriptor(sequence: np.ndarray) -> np.ndarray:
    """Return the 142-dimensional descriptor used for a six-channel scanpath."""
    values = np.asarray(sequence, dtype=float)
    if values.ndim != 2 or values.shape[1] < 6 or len(values) == 0:
        raise ValueError("A nonempty sequence with at least six channels is required")
    values = values[:, :6]
    n, channels = values.shape
    quantiles = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    time = np.linspace(-1.0, 1.0, n)
    output: list[float] = []
    for channel in range(channels):
        signal = values[:, channel]
        finite = signal[np.isfinite(signal)]
        fill = float(np.median(finite)) if len(finite) else 0.0
        signal = np.nan_to_num(signal, nan=fill, posinf=fill, neginf=fill)
        difference = np.diff(signal)
        centered = signal - signal.mean()
        scale = signal.std()
        skew = np.mean(centered**3) / scale**3 if scale > 0 and n > 2 else 0.0
        kurtosis = np.mean(centered**4) / scale**4 - 3.0 if scale > 0 and n > 3 else 0.0
        slope = float(np.dot(time, centered) / np.dot(time, time)) if n > 1 else 0.0
        thirds = np.array_split(signal, 3)
        output.extend(
            [
                float(signal.mean()),
                float(signal.std()),
                float(signal.min()),
                float(signal.max()),
                float(skew),
                float(kurtosis),
                *[float(value) for value in np.quantile(signal, quantiles)],
                *[float(part.mean()) for part in thirds],
                slope,
                float(np.mean(np.abs(difference))) if len(difference) else 0.0,
                float(np.std(difference)) if len(difference) else 0.0,
                _safe_corr(signal[:-1], signal[1:]) if n > 2 else 0.0,
            ]
        )
    for first in range(channels):
        for second in range(first + 1, channels):
            output.append(_safe_corr(values[:, first], values[:, second]))
    output.append(float(n))
    x = np.nan_to_num(values[:, 2], nan=np.nanmedian(values[:, 2]))
    y = np.nan_to_num(values[:, 3], nan=np.nanmedian(values[:, 3]))
    dx, dy = np.diff(x), np.diff(y)
    amplitude = np.hypot(dx, dy)
    output.extend(
        [
            float(np.mean(dx < 0)) if len(dx) else 0.0,
            float(np.mean(np.abs(dy) > np.nanstd(y))) if len(dy) else 0.0,
            float(amplitude.sum()),
            float(amplitude.mean()) if len(amplitude) else 0.0,
            float(np.quantile(amplitude, 0.9)) if len(amplitude) else 0.0,
            float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
        ]
    )
    result = np.nan_to_num(np.asarray(output, dtype=np.float32))
    if result.shape != (142,):
        raise AssertionError(f"Expected 142 descriptors, found {result.shape}")
    return result


def _load_sbsat_sequences(bundle: Path, metadata: pd.DataFrame) -> np.ndarray:
    root = bundle / "sbsat"
    values = np.load(root / "values.npy", mmap_mode="r")
    offsets = np.load(root / "offsets.npy")
    source = pd.read_csv(root / "sequence_metadata.csv")
    if len(offsets) != len(source) + 1:
        raise ValueError("SB-SAT offsets and sequence metadata have different lengths")
    lookup = {str(key): index for index, key in enumerate(source.unique_trial_id)}
    if len(lookup) != len(source):
        raise ValueError("SB-SAT sequence identifiers are not unique")
    cache: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    for trial in metadata.unique_trial_id.astype(str):
        if trial not in lookup:
            raise ValueError(f"Missing SB-SAT scanpath for {trial}")
        index = lookup[trial]
        key = (int(offsets[index]), int(offsets[index + 1]))
        if key not in cache:
            cache[key] = sequence_descriptor(values[key[0] : key[1], :6])
        rows.append(cache[key])
    return np.stack(rows)


def load_dataset(bundle: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load metadata and the feature views used in the primary analysis."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    metadata = _validate_metadata(
        pd.read_csv(bundle / "metadata" / f"{dataset}.csv"), dataset
    )
    metadata["material_id"] = material_ids(metadata, dataset)
    if dataset == "OneStop_RC":
        text_frame = pd.read_feather(bundle / "onestop" / "text_features.feather")
        gaze_frame = pd.read_feather(bundle / "onestop" / "gaze_features.feather")
        if text_frame.unique_trial_id.duplicated().any() or gaze_frame.unique_trial_id.duplicated().any():
            raise ValueError("OneStop feature identifiers are not unique")
        text_frame = text_frame.set_index("unique_trial_id")
        gaze_frame = gaze_frame.set_index("unique_trial_id")
        identifiers = metadata.unique_trial_id
        if not set(identifiers).issubset(text_frame.index) or not set(identifiers).issubset(gaze_frame.index):
            raise ValueError("OneStop feature tables do not cover every metadata row")
        text_columns = [column for column in text_frame if column.startswith("text_")]
        gaze_columns = [
            column
            for column in gaze_frame
            if column not in {"label", "unique_paragraph_id"}
        ]
        text = text_frame.loc[identifiers, text_columns].to_numpy(np.float32)
        gaze = gaze_frame.loc[identifiers, gaze_columns].to_numpy(np.float32)
        if text.shape[1] != 15 or gaze.shape[1] != 525:
            raise ValueError(f"Expected OneStop dimensions 15 and 525, found {text.shape[1]} and {gaze.shape[1]}")
        if "label" in gaze_frame:
            np.testing.assert_array_equal(
                gaze_frame.loc[identifiers, "label"].to_numpy(), metadata.label.to_numpy()
            )
        views = {"text": text, "gaze": gaze, "joint": np.c_[text, gaze].astype(np.float32)}
    else:
        scanpath = _load_sbsat_sequences(bundle, metadata)
        views = {"compact": scanpath[:, 80:120].copy(), "scanpath": scanpath}
    for name, matrix in views.items():
        if len(matrix) != len(metadata) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {dataset}/{name} feature matrix")
    return metadata, views


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", str(text).casefold()))


def onestop_representation_views(
    metadata: pd.DataFrame, joint: np.ndarray, qa_path: Path
) -> dict[str, np.ndarray]:
    lookup: dict[str, tuple[str, str]] = {}
    source = json.loads(qa_path.read_text())
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            for level in ("Adv", "Ele"):
                for question in paragraph["qas"]:
                    key = (
                        f"{article['article_id']}_{level}_{paragraph['paragraph_id']}_"
                        f"q{question['q_ind']}"
                    )
                    options = sorted(normalize_text(value) for value in question["answers"])
                    lookup[key] = (normalize_text(question["question"]), " ".join(options))
    items = np.array(sorted(lookup))
    if set(items) != set(metadata.item_key.astype(str)):
        raise ValueError("OneStop question file and evaluated items differ")
    vectorizer = HashingVectorizer(
        n_features=64,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    )
    question = vectorizer.transform([lookup[key][0] for key in items]).toarray().astype(np.float32)
    options = vectorizer.transform([lookup[key][1] for key in items]).toarray().astype(np.float32)
    index = {key: position for position, key in enumerate(items)}
    row = metadata.item_key.astype(str).map(index).to_numpy(int)
    return {
        "joint": joint.astype(np.float32),
        "question": np.c_[joint, question[row]].astype(np.float32),
        "question_options": np.c_[joint, question[row], options[row]].astype(np.float32),
    }


def _text_statistics(tokens: list[str]) -> list[float]:
    words = [word.lower() for word in tokens if word]
    if not words:
        raise ValueError("Text field has no lexical tokens")
    lengths = np.array([len(word) for word in words], float)
    surprisal = np.array(
        [-np.log2(word_frequency(word, "en", minimum=1e-11)) for word in words]
    )
    result = [float(len(words)), float(len(set(words)) / len(words))]
    for values in (lengths, surprisal):
        result.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(np.median(values)),
                float(np.quantile(values, 0.1)),
                float(np.quantile(values, 0.9)),
                float(values.min()),
                float(values.max()),
            ]
        )
    return result


def sbsat_lexical_descriptors(metadata: pd.DataFrame, stimuli_path: Path) -> np.ndarray:
    source = pd.read_csv(stimuli_path)
    passage = source[source.filename.str.startswith("reading-")].copy()
    last_word = passage.groupby("filename").word_number.transform("max")
    passage = passage[(passage.word_number > 3) & (passage.word_number <= last_word - 4)].copy()
    passage["page"] = passage.filename.str.extract(r"-(\d+)\.png$")[0].astype(int)
    passages: dict[str, list[str]] = {}
    for story, group in passage.sort_values(
        ["stimulus_type", "page", "word_number"]
    ).groupby("stimulus_type"):
        passages[str(story)] = re.findall(
            r"[A-Za-z]+(?:['’][A-Za-z]+)?", " ".join(group.word.fillna(""))
        )
    questions = source[source.filename.str.startswith("question-")][
        ["filename", "stimulus_type", "question"]
    ].drop_duplicates()
    questions["number"] = questions.filename.str.extract(r"-(\d+)\.png$")[0].astype(int)
    if len(questions) != 20 or questions.duplicated(["stimulus_type", "number"]).any():
        raise ValueError("Expected 20 distinct SB-SAT question texts")
    lookup: dict[str, list[float]] = {}
    for row in questions.itertuples():
        item = f"{row.stimulus_type}_q{row.number}"
        question_tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", row.question)
        passage_tokens = passages[str(row.stimulus_type)]
        question_types = set(word.lower() for word in question_tokens)
        passage_types = set(word.lower() for word in passage_tokens)
        overlap = len(question_types & passage_types) / len(question_types)
        lookup[item] = (
            _text_statistics(passage_tokens)
            + _text_statistics(question_tokens)
            + [float(overlap)]
        )
    if set(lookup) != set(metadata.item_key.astype(str)):
        raise ValueError("SB-SAT stimulus text and evaluated items differ")
    matrix = np.asarray([lookup[str(item)] for item in metadata.item_key], dtype=np.float32)
    if matrix.shape != (len(metadata), 33) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected an n-by-33 SB-SAT lexical matrix, found {matrix.shape}")
    return matrix


def extract_onestop_trial_order(
    archive: Path, metadata_path: Path, output: Path
) -> pd.DataFrame:
    """Extract participant-local order for the retained ordinary-reading trials."""
    metadata = pd.read_csv(metadata_path)
    universe = set(metadata.unique_trial_id.astype(str))
    columns = [
        "participant_id",
        "TRIAL_INDEX",
        "article_batch",
        "article_id",
        "difficulty_level",
        "paragraph_id",
        "repeated_reading_trial",
    ]
    parts = []
    with zipfile.ZipFile(archive) as zipped:
        candidates = [
            name
            for name in zipped.namelist()
            if name.endswith("ia_Paragraph_ordinary.csv")
            and not name.startswith("__MACOSX")
        ]
        if len(candidates) != 1:
            raise ValueError("Expected one ia_Paragraph_ordinary.csv in the archive")
        with zipped.open(candidates[0]) as stream:
            for chunk in pd.read_csv(
                stream, usecols=columns, chunksize=250_000, low_memory=False
            ):
                chunk["unique_trial_id"] = (
                    chunk.participant_id.astype(str)
                    + "_"
                    + chunk.article_batch.astype(int).astype(str)
                    + "_"
                    + chunk.article_id.astype(int).astype(str)
                    + "_"
                    + chunk.difficulty_level.astype(str)
                    + "_"
                    + chunk.paragraph_id.astype(int).astype(str)
                    + "_0_0"
                )
                part = chunk.loc[
                    chunk.unique_trial_id.isin(universe),
                    ["unique_trial_id", "participant_id", "TRIAL_INDEX"],
                ]
                parts.append(part.drop_duplicates())
    order = pd.concat(parts, ignore_index=True).drop_duplicates()
    if set(order.unique_trial_id.astype(str)) != universe:
        raise ValueError("Raw trial order does not cover the retained metadata")
    if order.unique_trial_id.duplicated().any():
        raise ValueError("A retained trial has multiple order values")
    if order.duplicated(["participant_id", "TRIAL_INDEX"]).any():
        raise ValueError("Trial order is tied within a participant")
    order = order.rename(columns={"TRIAL_INDEX": "trial_index"}).sort_values(
        ["participant_id", "trial_index"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    order.to_csv(output, index=False)
    return order.reset_index(drop=True)
