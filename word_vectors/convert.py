"""Convert between word vector formats.

We provide the main :py:func:`~word_vectors.convert.convert` function for converting
between arbitrary formats based on the passed :py:attr:`~word_vectors.FileType` (or
by sniffing the input file with :py:func:`~word_vectors.read.sniff` when not provided)
as well as several convenience function for converting between different pairs of formats.
"""

from typing import Union, TextIO, BinaryIO, Optional
from word_vectors import FileType
from word_vectors.read import read
from word_vectors.write import write
from word_vectors.utils import find_max, create_output_path


# We don't know what mode to open the file in (text for things like Glove while
# binary for things like Word2Vec or Dense) we can't use the `@file_or_name`
# decorator directly but all the functions we call use that so we can handle
# all the file formats.
def convert(
    f: Union[str, TextIO, BinaryIO],
    output: Optional[str] = None,
    output_file_type: FileType = FileType.DENSE,
    input_file_type: Optional[FileType] = None,
):
    """Convert vectors from one format to another.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
        output_file_type: The vector serialization format to use when
            writing out the vectors.
        input_file_type: An explicit vector format to use when reading.
    """
    w, wv = read(f, input_file_type)
    len_ = find_max(w.keys())
    output = create_output_path(f, output_file_type) if output is None else output
    write(output, w, wv, output_file_type, len_)


def w2v_to_dense(f: Union[str, BinaryIO], output: Optional[str] = None):
    """Convert binary Word2Vec formatted vectors to the Dense format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.DENSE, FileType.W2V)


def glove_to_dense(f: Union[str, TextIO], output: Optional[str] = None):
    """Convert GloVe formatted vectors to the Dense format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.Dense, FileType.GLOVE)


def w2v_text_to_dense(f: Union[str, TextIO], output: Optional[str] = None):
    """Convert text Word2Vec formatted vectors to the Dense format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.Dense, FileType.W2V_TEXT)


def w2v_to_w2v_text(f: Union[str, BinaryIO], output: Optional[str] = None):
    """Convert binary Word2Vec formatted vectors to the Binary Word2Vec format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.W2V_TEXT, FileType.W2V)


def w2v_to_glove(f: Union[str, BinaryIO], output: Optional[str] = None):
    """Convert binary Word2Vec formatted vectors to the GloVe format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.GLOVE, FileType.W2V)


def w2v_text_to_glove(f: Union[str, TextIO], output: Optional[str] = None):
    """Convert text Word2Vec formatted vectors to the Glove format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.GLOVE, FileType.W2V_TEXT)


def w2v_text_to_w2v(f: Union[str, TextIO], output: Optional[str] = None):
    """Convert text Word2Vec formatted vectors to the binary Word2Vec format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.W2V, FileType.W2V_TEXT)


def glove_to_w2v(f: Union[str, TextIO], output: Optional[str] = None):
    """Convert GloVe formatted vectors to the binary Word2Vec format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.W2V, FileType.GLOVE)


def glove_to_w2v_text(f: Union[str, TextIO], output: Optional[str] = None):
    """Convert GloVe formatted vectors to the text Word2Vec format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.W2V_TEXT, FileType.GLOVE)


def dense_to_w2v(f: Union[str, BinaryIO], output: Optional[str] = None):
    """Convert Dense formatted vectors to the binary Word2Vec format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.W2V, FileType.DENSE)


def dense_to_w2v_text(f: Union[str, BinaryIO], output: Optional[str] = None):
    """Convert Dense formatted vectors to the text Word2Vec format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    convert(f, output, FileType.W2V_TEXT, FileType.DENSE)


def dense_to_glove(f: Union[str, BinaryIO], output: Optional[str] = None):
    """Convert Dense formatted vectors to the GloVe format.

    Args:
        f: The file to read from.
        output: The name for the output file. If not provided we use the
            input file name with a modified extension.
    """
    covert(f, output, FileType.GLOVE, FileType.DENSE)
