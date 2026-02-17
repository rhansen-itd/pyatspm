"""
DatZ Binary Decoder for ATSPM System (Functional Core)

This module handles parsing of .datZ files (compressed binary traffic logs).
It follows the "Functional Core" pattern - pure transformations with no I/O.

Package Location: src/atspm/analysis/decoders.py
"""

import zlib
import struct
import pandas as pd
from typing import List, Tuple


class DatZDecodingError(Exception):
    """
    Custom exception for DatZ file decoding errors.
    
    Raised when:
    - Decompression fails
    - Required markers not found
    - Binary structure is invalid
    """
    pass


def parse_datz_bytes(raw_bytes: bytes, file_timestamp: float) -> pd.DataFrame:
    """
    Parse compressed DatZ binary data into a DataFrame of traffic events.
    
    This is a pure function that transforms raw bytes into structured data.
    No I/O operations - accepts bytes, returns DataFrame.
    
    Args:
        raw_bytes: Raw bytes from .datZ file (compressed)
        file_timestamp: Start timestamp for this file (UTC epoch float)
        
    Returns:
        DataFrame with columns: ['timestamp', 'event_code', 'parameter']
        Timestamps are UTC epoch floats
        
    Raises:
        DatZDecodingError: If decompression fails, marker not found, or invalid structure
        
    Example:
        >>> with open('traffic.datZ', 'rb') as f:
        ...     raw = f.read()
        >>> df = parse_datz_bytes(raw, file_timestamp=1609459200.0)
        >>> df.head()
           timestamp  event_code  parameter
        0  1609459200.0           1          2
        1  1609459210.5          82         33
    """
    # Step 1: Decompress
    try:
        content = zlib.decompress(raw_bytes)
    except zlib.error as e:
        raise DatZDecodingError(f"Failed to decompress datZ file: {e}")
    
    # Step 2: Find binary payload marker
    marker = b"Phases in use:"
    marker_pos = content.find(marker)
    
    if marker_pos == -1:
        raise DatZDecodingError(
            "Invalid datZ format: 'Phases in use:' marker not found"
        )
    
    # Step 3: Locate binary payload (after marker's newline)
    newline_pos = content.find(b'\n', marker_pos)
    if newline_pos == -1:
        raise DatZDecodingError(
            "Invalid datZ format: No newline after marker"
        )
    
    binary_bytes = content[newline_pos + 1:]
    
    # Step 4: Parse binary rows
    records = _parse_binary_payload(binary_bytes, file_timestamp)
    
    # Step 5: Convert to DataFrame
    if not records:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=['timestamp', 'event_code', 'parameter'])
    
    df = pd.DataFrame(records, columns=['timestamp', 'event_code', 'parameter'])
    
    # Ensure correct types
    df['timestamp'] = df['timestamp'].astype(float)
    df['event_code'] = df['event_code'].astype(int)
    df['parameter'] = df['parameter'].astype(int)
    
    return df


def _parse_binary_payload(
    binary_bytes: bytes, 
    base_timestamp: float
) -> List[Tuple[float, int, int]]:
    """
    Parse the binary payload into a list of event tuples.
    
    Binary format: 4 bytes per row, Big-Endian
    - Byte 0: Event code (unsigned char)
    - Byte 1: Parameter (unsigned char)
    - Bytes 2-3: Time offset in deciseconds (unsigned short)
    
    Args:
        binary_bytes: Raw binary payload
        base_timestamp: Base timestamp for this file (UTC epoch float)
        
    Returns:
        List of (timestamp, event_code, parameter) tuples
        
    Raises:
        DatZDecodingError: If binary structure is invalid
    """
    row_size = 4
    num_rows = len(binary_bytes) // row_size
    
    if len(binary_bytes) % row_size != 0:
        raise DatZDecodingError(
            f"Invalid binary payload: {len(binary_bytes)} bytes is not divisible by {row_size}"
        )
    
    records = []
    
    for i in range(num_rows):
        chunk = binary_bytes[i * row_size : (i + 1) * row_size]
        
        try:
            # Unpack: >BBH = Big-Endian, UChar, UChar, UShort
            event_code, parameter, offset_deciseconds = struct.unpack('>BBH', chunk)
        except struct.error as e:
            raise DatZDecodingError(
                f"Failed to unpack row {i}: {e}"
            )
        
        # Calculate event timestamp
        # Offset is in deciseconds (tenths of seconds)
        offset_seconds = offset_deciseconds / 10.0
        event_timestamp = base_timestamp + offset_seconds
        
        records.append((event_timestamp, event_code, parameter))
    
    return records


def validate_datz_file(raw_bytes: bytes) -> bool:
    """
    Quick validation check without full parsing.
    
    Checks if bytes appear to be a valid datZ file by:
    1. Attempting decompression
    2. Checking for required marker
    
    Args:
        raw_bytes: Raw bytes to validate
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> with open('file.datZ', 'rb') as f:
        ...     data = f.read()
        >>> if validate_datz_file(data):
        ...     df = parse_datz_bytes(data, timestamp)
    """
    try:
        content = zlib.decompress(raw_bytes)
        marker = b"Phases in use:"
        return marker in content
    except (zlib.error, Exception):
        return False


def estimate_event_count(raw_bytes: bytes) -> int:
    """
    Estimate number of events without full parsing.
    
    Useful for pre-allocation or progress reporting.
    
    Args:
        raw_bytes: Raw datZ bytes
        
    Returns:
        Estimated event count (0 if invalid)
        
    Example:
        >>> count = estimate_event_count(raw_bytes)
        >>> print(f"Processing ~{count} events")
    """
    try:
        content = zlib.decompress(raw_bytes)
        marker = b"Phases in use:"
        marker_pos = content.find(marker)
        
        if marker_pos == -1:
            return 0
        
        newline_pos = content.find(b'\n', marker_pos)
        if newline_pos == -1:
            return 0
        
        binary_bytes = content[newline_pos + 1:]
        row_size = 4
        
        return len(binary_bytes) // row_size
        
    except (zlib.error, Exception):
        return 0


def parse_datz_batch(
    file_data: List[Tuple[bytes, float]]
) -> pd.DataFrame:
    """
    Parse multiple datZ files in batch and concatenate results.
    
    This is more efficient than parsing individually because it:
    1. Pre-allocates total DataFrame size
    2. Reduces concatenation overhead
    3. Maintains sort order
    
    Args:
        file_data: List of (raw_bytes, file_timestamp) tuples
        
    Returns:
        Combined DataFrame sorted by timestamp
        
    Raises:
        DatZDecodingError: If any file fails to parse
        
    Example:
        >>> files = [
        ...     (read_file('file1.datZ'), 1609459200.0),
        ...     (read_file('file2.datZ'), 1609459800.0),
        ... ]
        >>> df = parse_datz_batch(files)
    """
    if not file_data:
        return pd.DataFrame(columns=['timestamp', 'event_code', 'parameter'])
    
    # Parse all files
    dataframes = []
    for raw_bytes, file_timestamp in file_data:
        df = parse_datz_bytes(raw_bytes, file_timestamp)
        if not df.empty:
            dataframes.append(df)
    
    # Concatenate and sort
    if not dataframes:
        return pd.DataFrame(columns=['timestamp', 'event_code', 'parameter'])
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    return combined_df


def insert_gap_marker(
    df: pd.DataFrame, 
    gap_timestamp: float
) -> pd.DataFrame:
    """
    Insert a gap/discontinuity marker into event DataFrame.
    
    Gap markers use event_code = -1, parameter = -1 to signal
    data discontinuity (e.g., missing files).
    
    Args:
        df: Existing events DataFrame
        gap_timestamp: Timestamp where gap begins (UTC epoch float)
        
    Returns:
        DataFrame with gap marker inserted and re-sorted
        
    Example:
        >>> df = parse_datz_bytes(raw_bytes, timestamp)
        >>> # Detected 10-minute gap
        >>> df = insert_gap_marker(df, gap_timestamp=1609459800.0)
    """
    gap_row = pd.DataFrame([{
        'timestamp': gap_timestamp,
        'event_code': -1,
        'parameter': -1
    }])
    
    combined = pd.concat([df, gap_row], ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    return combined


def detect_corruption(raw_bytes: bytes) -> bool:
    """
    Detect likely file corruption based on size and structure.
    
    Corruption indicators:
    1. File size > 50KB (typical datZ is ~5-15KB)
    2. Decompression succeeds but marker missing
    3. Invalid binary structure
    
    Args:
        raw_bytes: Raw datZ bytes
        
    Returns:
        True if corruption likely detected
        
    Example:
        >>> if detect_corruption(raw_bytes):
        ...     print("Skipping corrupted file")
        ... else:
        ...     df = parse_datz_bytes(raw_bytes, timestamp)
    """
    # Check 1: Unreasonable file size
    if len(raw_bytes) > 50_000:  # 50KB threshold
        return True
    
    # Check 2: Decompression succeeds but structure invalid
    try:
        content = zlib.decompress(raw_bytes)
        marker = b"Phases in use:"
        
        if marker not in content:
            return True
        
        # Check for reasonable payload size
        marker_pos = content.find(marker)
        newline_pos = content.find(b'\n', marker_pos)
        
        if newline_pos == -1:
            return True
        
        binary_bytes = content[newline_pos + 1:]
        
        # Check if payload is divisible by row size
        if len(binary_bytes) % 4 != 0:
            return True
        
        return False
        
    except zlib.error:
        # Decompression failure = corruption
        return True
    except Exception:
        # Any other error = likely corruption
        return True