import argparse
from word_vectors import FileType
from word_vectors.convert import convert


def main():
    parser = argparse.ArgumentParser(description="Convert Pre-trained embeddings between different formats")
    parser.add_argument("embeddings")
    parser.add_argument("--output-format", "--output_format", default=FileType.LEADER, type=FileType.from_string)
    parser.add_argument("--input-format", "--input_format", type=FileType.from_string)
    parser.add_argument("--output", help="The output path.")
    args = parser.parse_args()

    convert(args.embeddings, output=args.output, output_file_type=args.output_format, input_file_type=args.input_format)


if __name__ == "__main__":
    main()
