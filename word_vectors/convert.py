"""Convert word vector formats."""

import pathlib
from typing import Union, TextIO, BinaryIO, Optional
from word_vectors.read import read
from word_vectors.utils import find_max
from word_vectors.write import write_dense


# We don't know what mode to open the file in (text for things like Glove while
# binary for things like Word2Vec or Dense) we can't use the `@file_or_name`
# decorator directly but all the functions we call use that so we can handle
# all the file formats.
def convert(f: Union[str, TextIO, BinaryIO], output: Optional[str] = None):
    """Convert vectors to the dense format.

    If an output target is not provided it is created by appending ``.dense``
    to the input file name.

    Args:
        f: The file to read from.
        output: The name of the output file.
    """
    w, wv = read(f)
    len_ = find_max(w.keys())
    if output is None:
        if isinstance(f, (str, pathlib.PurePath)):
            output = str(f)
        else:
            output = f.name
        output = f"{output}.dense"
    write_dense(output, w, wv, len_)
