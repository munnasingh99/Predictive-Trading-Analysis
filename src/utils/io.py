"""
Input/Output utility module for predictive trading signals.

This module provides various I/O utilities for the trading system including:
- File operations and path management
- Data serialization and deserialization
- CSV/Excel file handling
- JSON/Pickle utilities
- Data validation and cleaning
- Progress tracking for long operations

Key features:
- Safe file operations with error handling
- Data format conversion utilities
- Progress bars for data processing
- Temporary file management
- Archive and compression utilities
"""

import json
import logging
import pickle
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FileManager:
    """File operations and path management utilities."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize file manager.

        Args:
            base_path: Base path for file operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.temp_dir = self.base_path / "temp"

    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary.

        Args:
            path: Directory path

        Returns:
            Path object for the directory
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def safe_file_operation(self, operation_func, *args, **kwargs) -> Any:
        """Safely execute file operation with error handling.

        Args:
            operation_func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result or None if failed
        """
        try:
            return operation_func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return None
        except PermissionError as e:
            logger.error(f"Permission denied: {e}")
            return None
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return None

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get file information.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information
        """
        path = Path(file_path)

        if not path.exists():
            return {"exists": False}

        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "extension": path.suffix.lower(),
            "name": path.name,
            "parent": str(path.parent)
        }

    def list_files(self, directory: Union[str, Path], pattern: str = "*",
                   recursive: bool = False) -> List[Path]:
        """List files in directory matching pattern.

        Args:
            directory: Directory to search
            pattern: File pattern (glob style)
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        try:
            if recursive:
                return list(dir_path.rglob(pattern))
            else:
                return list(dir_path.glob(pattern))
        except Exception as e:
            logger.error(f"Failed to list files in {directory}: {e}")
            return []

    def copy_file(self, source: Union[str, Path], destination: Union[str, Path],
                  overwrite: bool = False) -> bool:
        """Copy file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file

        Returns:
            True if successful, False otherwise
        """
        source_path = Path(source)
        dest_path = Path(destination)

        if not source_path.exists():
            logger.error(f"Source file does not exist: {source}")
            return False

        if dest_path.exists() and not overwrite:
            logger.warning(f"Destination file exists and overwrite=False: {destination}")
            return False

        try:
            self.ensure_directory(dest_path.parent)
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False

    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Move file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)

            self.ensure_directory(dest_path.parent)
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"Moved {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to move file: {e}")
            return False

    def delete_file(self, file_path: Union[str, Path], confirm: bool = True) -> bool:
        """Delete file.

        Args:
            file_path: Path to file to delete
            confirm: Whether file must exist (False = no error if missing)

        Returns:
            True if successful or file doesn't exist and confirm=False
        """
        path = Path(file_path)

        if not path.exists():
            if confirm:
                logger.error(f"File does not exist: {file_path}")
                return False
            else:
                return True

        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            logger.info(f"Deleted {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    def create_temp_file(self, suffix: str = "", prefix: str = "temp_") -> Path:
        """Create temporary file.

        Args:
            suffix: File suffix
            prefix: File prefix

        Returns:
            Path to temporary file
        """
        import tempfile

        self.ensure_directory(self.temp_dir)

        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=self.temp_dir
        )

        # Close file descriptor
        import os
        os.close(fd)

        return Path(temp_path)

    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up old temporary files.

        Args:
            max_age_hours: Maximum age in hours before deletion

        Returns:
            Number of files deleted
        """
        if not self.temp_dir.exists():
            return 0

        deleted_count = 0
        current_time = datetime.now()

        for temp_file in self.temp_dir.iterdir():
            try:
                file_age = current_time - datetime.fromtimestamp(temp_file.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    if temp_file.is_file():
                        temp_file.unlink()
                    else:
                        shutil.rmtree(temp_file)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} temporary files")

        return deleted_count


class DataSerializer:
    """Data serialization and deserialization utilities."""

    @staticmethod
    def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
        """Save data to JSON file.

        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation

        Returns:
            True if successful
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str, ensure_ascii=False)

            logger.info(f"Saved JSON data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return False

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Optional[Any]:
        """Load data from JSON file.

        Args:
            file_path: Input file path

        Returns:
            Loaded data or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            return None

    @staticmethod
    def save_pickle(data: Any, file_path: Union[str, Path]) -> bool:
        """Save data using pickle.

        Args:
            data: Data to save
            file_path: Output file path

        Returns:
            True if successful
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved pickle data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle: {e}")
            return False

    @staticmethod
    def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
        """Load data from pickle file.

        Args:
            file_path: Input file path

        Returns:
            Loaded data or None if failed
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded pickle data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load pickle: {e}")
            return None


class CSVHandler:
    """CSV file handling utilities."""

    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: Union[str, Path],
                 index: bool = False, **kwargs) -> bool:
        """Save DataFrame to CSV.

        Args:
            df: DataFrame to save
            file_path: Output file path
            index: Whether to save index
            **kwargs: Additional pandas to_csv arguments

        Returns:
            True if successful
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(path, index=index, **kwargs)
            logger.info(f"Saved CSV with {len(df)} rows to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return False

    @staticmethod
    def load_csv(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """Load CSV file as DataFrame.

        Args:
            file_path: Input file path
            **kwargs: Additional pandas read_csv arguments

        Returns:
            DataFrame or None if failed
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Loaded CSV with {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return None

    @staticmethod
    def validate_csv_format(file_path: Union[str, Path],
                           required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate CSV file format.

        Args:
            file_path: CSV file path
            required_columns: List of required column names

        Returns:
            Validation results dictionary
        """
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {}
        }

        try:
            # Try to read just the header
            df_header = pd.read_csv(file_path, nrows=0)
            columns = df_header.columns.tolist()

            result["info"]["columns"] = columns
            result["info"]["column_count"] = len(columns)

            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(columns)
                if missing_columns:
                    result["errors"].append(f"Missing required columns: {list(missing_columns)}")
                else:
                    result["info"]["required_columns_present"] = True

            # Try to read a sample
            df_sample = pd.read_csv(file_path, nrows=100)
            result["info"]["sample_rows"] = len(df_sample)

            # Check for empty DataFrame
            if df_sample.empty:
                result["errors"].append("CSV file is empty")

            # Check for missing values
            missing_pct = df_sample.isnull().sum() / len(df_sample)
            high_missing = missing_pct[missing_pct > 0.5]
            if not high_missing.empty:
                result["warnings"].append(f"Columns with >50% missing values: {list(high_missing.index)}")

            # Success if no errors
            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["errors"].append(f"Failed to read CSV file: {e}")

        return result

    @staticmethod
    def clean_csv_data(df: pd.DataFrame, date_columns: Optional[List[str]] = None,
                      numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Clean CSV data.

        Args:
            df: Input DataFrame
            date_columns: List of date column names
            numeric_columns: List of numeric column names

        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()

        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col in cleaned_df.columns:
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                        logger.debug(f"Converted {col} to datetime")
                    except Exception as e:
                        logger.warning(f"Failed to convert {col} to datetime: {e}")

        # Convert numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in cleaned_df.columns:
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                        logger.debug(f"Converted {col} to numeric")
                    except Exception as e:
                        logger.warning(f"Failed to convert {col} to numeric: {e}")

        # Remove completely empty rows and columns
        cleaned_df = cleaned_df.dropna(how='all', axis=0)
        cleaned_df = cleaned_df.dropna(how='all', axis=1)

        return cleaned_df


class ArchiveManager:
    """Archive and compression utilities."""

    @staticmethod
    def create_zip_archive(source_path: Union[str, Path], archive_path: Union[str, Path],
                          include_pattern: str = "*") -> bool:
        """Create ZIP archive from directory.

        Args:
            source_path: Source directory path
            archive_path: Output archive path
            include_pattern: Pattern for files to include

        Returns:
            True if successful
        """
        try:
            source = Path(source_path)
            archive = Path(archive_path)

            archive.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if source.is_file():
                    zipf.write(source, source.name)
                else:
                    for file_path in source.rglob(include_pattern):
                        if file_path.is_file():
                            arcname = file_path.relative_to(source)
                            zipf.write(file_path, arcname)

            logger.info(f"Created ZIP archive: {archive_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create ZIP archive: {e}")
            return False

    @staticmethod
    def extract_zip_archive(archive_path: Union[str, Path], extract_path: Union[str, Path]) -> bool:
        """Extract ZIP archive.

        Args:
            archive_path: Archive file path
            extract_path: Extraction directory path

        Returns:
            True if successful
        """
        try:
            archive = Path(archive_path)
            extract_dir = Path(extract_path)

            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(archive, 'r') as zipf:
                zipf.extractall(extract_dir)

            logger.info(f"Extracted ZIP archive to: {extract_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract ZIP archive: {e}")
            return False


class ProgressTracker:
    """Progress tracking utilities for long operations."""

    @staticmethod
    def create_progress_bar(total: int, description: str = "Processing") -> tqdm:
        """Create progress bar.

        Args:
            total: Total number of items
            description: Progress description

        Returns:
            tqdm progress bar instance
        """
        return tqdm(total=total, desc=description, unit="items")

    @staticmethod
    def process_with_progress(items: List[Any], process_func, description: str = "Processing",
                            **kwargs) -> List[Any]:
        """Process items with progress bar.

        Args:
            items: List of items to process
            process_func: Function to process each item
            description: Progress description
            **kwargs: Additional arguments for process_func

        Returns:
            List of processed results
        """
        results = []

        with tqdm(total=len(items), desc=description) as pbar:
            for item in items:
                try:
                    result = process_func(item, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item {item}: {e}")
                    results.append(None)
                pbar.update(1)

        return results

    @staticmethod
    def save_with_progress(df: pd.DataFrame, file_path: Union[str, Path],
                          chunk_size: int = 10000, **kwargs) -> bool:
        """Save large DataFrame with progress tracking.

        Args:
            df: DataFrame to save
            file_path: Output file path
            chunk_size: Size of chunks for processing
            **kwargs: Additional arguments for to_csv

        Returns:
            True if successful
        """
        try:
            total_chunks = (len(df) - 1) // chunk_size + 1

            with tqdm(total=total_chunks, desc="Saving CSV") as pbar:
                for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
                    chunk_end = min(chunk_start + chunk_size, len(df))
                    chunk_df = df.iloc[chunk_start:chunk_end]

                    # Write header only for first chunk
                    header = i == 0
                    mode = 'w' if i == 0 else 'a'

                    chunk_df.to_csv(file_path, mode=mode, header=header, index=False, **kwargs)
                    pbar.update(1)

            logger.info(f"Saved large DataFrame ({len(df)} rows) to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save large DataFrame: {e}")
            return False


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    try:
        return Path(file_path).stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def ensure_path_exists(path: Union[str, Path]) -> Path:
    """Ensure path exists, create if necessary.

    Args:
        path: Path to ensure

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
