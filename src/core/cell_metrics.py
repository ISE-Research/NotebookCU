import collections
import hashlib
import json
import logging
import os
import re
import statistics as sts

import pandas as pd
from radon.visitors import ComplexityVisitor
from tqdm import tqdm

from utils.config import CACHE_PATH

logger = logging.getLogger(__name__)

tqdm.pandas()

PYTHON_KEYWORDS = [
    "True",
    "False",
    "None",
    "and",
    "as",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
]

PYTHON_OPERATORS = [
    "+",
    "-",
    "*",
    "/",
    "**",
    "%",
    "//",
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "//=",
    "**=",
    "&=",
    "|=",
    "^=",
    ">>=",
    "<<=",
    "==",
    "!=",
    ">=",
    "<=",
    ">",
    "<",
    " and ",
    " or ",
    " not ",
    " is ",
    " is not ",
    " in ",
    " not in ",
    "&",
    "|",
    "^",
    "~",
    "<<",
    ">>",
]

############## CODE METRICS ##############

# Helper Methods for Code Metrics


# OPERATOR: Number of Operators
def extract_operator_count(text: str) -> int:
    count = 0
    for operator in PYTHON_OPERATORS:
        count += text.count(operator)
    return count


# UOPERATOR: Number of Unique Operators
def extract_unique_operator_count(text: str) -> int:
    unique_count = 0
    for operator in PYTHON_OPERATORS:
        if operator in text:
            unique_count += 1
    return unique_count


# ID: Number of Identifiers
def extract_identifier_count(text: str) -> int:
    count = 0
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    for identifier in identifiers:
        if identifier not in PYTHON_KEYWORDS:
            count += 1
    return count


# ALID: Average Length of Identifiers
def extract_avg_len_identifier(text: str) -> int:
    identifier_lengths = []
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    for identifier in identifiers:
        if identifier not in PYTHON_KEYWORDS:
            identifier_lengths.append(len(identifier))
    if len(identifier_lengths) == 0:
        return 0
    return sum(identifier_lengths) // len(identifier_lengths)


# MLID: Max Length of Identifiers
def extract_max_len_identifier(text: str) -> int:
    identifier_lengths = []
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    for identifier in identifiers:
        if identifier not in PYTHON_KEYWORDS:
            identifier_lengths.append(len(identifier))
    if len(identifier_lengths) == 0:
        return 0
    return max(identifier_lengths)


# KLCID: Key Lines of Code with Identifier
def extract_klcid(text: str) -> int:
    unique_code_lines = set(text.split("\n"))
    identifier_count = []
    for line in unique_code_lines:
        identifier_count.append(extract_identifier_count(line))
    unique_lines_code_with_identifier = []
    for line in unique_code_lines:
        if extract_identifier_count(line) > 0:
            unique_lines_code_with_identifier.append(line)
    if len(unique_lines_code_with_identifier) == 0:
        return 0
    return sum(identifier_count) / len(unique_lines_code_with_identifier)


# OPRND: Number of Operands
def extract_operand_count(text: str) -> int:
    res = re.split(
        r"""[\-()\+\*\/\=\&\|\^\~\<\>\%]|
    \*\*|\/\/|
    \>\>|\<\<|
    \+\=|\-\=|\*\=|\/\=|\%\=|
    \/\/\=|\*\*\=
    \&\=|\|\=|\^\=|\>\>\=|\>\>\=|
    \=\=|\!\=|\>\=""",
        text,
    )
    res = list(filter(None, res))
    return len(res)


# UOPRND: Number of Unique Operands
def extract_unique_operand_count(text: str) -> int:
    res = re.split(
        r"""[\-()\+\*\/\=\&\|\^\~\<\>\%]|
    \*\*|\/\/|
    \>\>|\<\<|
    \+\=|\-\=|\*\=|\/\=|\%\=|
    \/\/\=|\*\*\=
    \&\=|\|\=|\^\=|\>\>\=|\>\>\=|
    \=\=|\!\=|\>\=""",
        text,
    )
    res = set(filter(None, res))
    return len(res)


# P: Number of Python Parameters
def extract_python_parameters_count(text: str) -> int:
    args = []
    args_regex = re.compile(
        r"""(
                [a-zA-Z_][a-zA-Z0-9_]*\((.*?)\)
            )""",
        flags=re.DOTALL | re.VERBOSE | re.MULTILINE,
    )
    try:
        s = re.findall(args_regex, text)
        z = [i[0] for i in s]
        for i in z:
            args = args + re.search(r"\((.*?)\)", i).group(1).split(",")
        return len(args)
    except:
        return 0


# Count the number of loop statements (while and for) in the given text
def extract_loop_statements_count(text: str) -> int:
    loop_keywords = ["while", "for"]
    count = 0
    for keyword in loop_keywords:
        count += text.count(keyword)
    return count


# Count the number of if statements in the given text
def extract_if_statements_count(text: str) -> int:
    if_keywords = ["if"]
    count = 0
    for keyword in if_keywords:
        count += text.count(keyword)
    return count


# S: Number of Python Statements
def extract_statements_count(text: str) -> int:
    return extract_loop_statements_count(text) + extract_if_statements_count(text)


# NBD: Nested Block Depth
def extract_nested_block_depth(text: str) -> float:
    blocks = text.split("\n")
    depth = []
    for block in blocks:
        depth.append(block.count(" "))
    return sum(depth) / len(depth)


# CyC: Cyclomatic Complexity
def extract_complexity_analysis(text: str) -> float:
    source = text.replace("%matplotlib inline", "")
    try:
        v = ComplexityVisitor.from_code(source)
        return v.complexity
    except Exception as e:
        # TODO: log error
        return 0


# Capture and extract import statements from the given source code
def capture_imports(text: str) -> list:
    import_regex = r"^\s*(?:from|import)\s+(\w+(?:\s*,\s*\w+)*)"
    # Find all import statements
    import_matches = re.findall(import_regex, text, re.MULTILINE)
    for i in import_matches:
        if "," in i:
            new_i = i.replace(" ", "").split(",")
            import_matches.extend(new_i)
            import_matches.remove(i)
    return import_matches


def get_list_of_all_api(code_df: pd.DataFrame) -> list:
    code_df["API"] = code_df["source"].apply(lambda x: capture_imports(str(x)))
    return [item for sublist in code_df["API"].values.tolist() for item in sublist]


def get_api_popularity_dict(api_list: list) -> dict:
    eap_dict = dict(collections.Counter(api_list))
    max_freq = eap_dict["sklearn"]
    eap_score_dict = dict()
    for k, v in eap_dict.items():
        eap_score_dict[k] = v / max_freq
    return eap_score_dict


# EAP: External api
def extract_eap_score(api_set: set, eap_score_dict: dict) -> float:
    score = 0
    for i in api_set:
        score += eap_score_dict.get(i, 0)
    return score


def extract_code_metrics(code_df: pd.DataFrame, eap_score_dict: dict) -> pd.DataFrame:
    code_df["API"] = code_df["source"].apply(lambda x: capture_imports(str(x)))
    code_df["EAP"] = code_df["API"].progress_apply(
        lambda x: extract_eap_score(api_set=set(x), eap_score_dict=eap_score_dict)
    )
    code_df["LOC"] = code_df["source"].apply(lambda x: x.count("\n") + 1 if type(x) == str else 0)
    code_df["BLC"] = code_df["LOC"].apply(lambda x: 1 if x == 0 else 0)
    code_df["UDF"] = code_df["source"].apply(
        lambda x: (sum([len(re.findall("^(?!#).*def ", y)) for y in x.split("\n")]) if type(x) == str else 0)
    )
    code_df["I"] = code_df["source"].apply(lambda x: x.count("import ") if type(x) == str else 0)
    code_df["EH"] = code_df["source"].apply(lambda x: x.count("try:") if type(x) == str else 0)
    code_df["ALLC"] = code_df["source"].apply(
        lambda x: sts.mean([len(y) for y in x.split("\n")]) if type(x) == str else 0
    )
    code_df["NVD"] = code_df["output_type"].progress_apply(lambda x: x.count("display_data") if type(x) == str else 0)
    code_df["NEC"] = code_df["output_type"].progress_apply(lambda x: x.count("execute_result") if type(x) == str else 0)
    code_df["S"] = code_df["source"].progress_apply(lambda x: extract_statements_count(str(x)))
    code_df["P"] = code_df["source"].progress_apply(lambda x: extract_python_parameters_count(str(x)))
    code_df["KLCID"] = code_df["source"].progress_apply(lambda x: extract_klcid(str(x)))
    code_df["NBD"] = code_df["source"].progress_apply(lambda x: extract_nested_block_depth(str(x)))
    code_df["OPRATOR"] = code_df["source"].progress_apply(lambda x: extract_operator_count(str(x)))
    code_df["OPRND"] = code_df["source"].progress_apply(lambda x: extract_operand_count(str(x)))
    code_df["UOPRND"] = code_df["source"].progress_apply(lambda x: extract_unique_operand_count(str(x)))
    code_df["UOPRATOR"] = code_df["source"].progress_apply(lambda x: extract_unique_operator_count(str(x)))
    code_df["ID"] = code_df["source"].progress_apply(lambda x: extract_identifier_count(str(x)))
    code_df["ALID"] = code_df["source"].progress_apply(lambda x: extract_avg_len_identifier(str(x)))
    code_df["MLID"] = code_df["source"].progress_apply(lambda x: extract_max_len_identifier(str(x)))
    code_df["CyC"] = code_df["source"].progress_apply(lambda x: extract_complexity_analysis(str(x)))
    code_df = extract_comment_metrics(code_df)

    columns_to_drop = ["source", "output_type", "execution_count"]
    code_df.drop(columns=columns_to_drop, inplace=True)
    return code_df


############## COMMENT METRICS ##############

# Helper Methods for Comment Metrics


# Count the number of inline comments in the given string
def count_inline_comment(string: str) -> str:
    inline_regex = re.compile(
        r"""(
            (?<=\#).+ # comments like: # This is a comment
        )""",
        flags=re.VERBOSE,
    )
    return len(re.findall(inline_regex, string))


# Extract and count the number of multi-line comments in the given string
def multi_line_comments(string: str) -> str:

    # Python comments
    multi_line_python_regex = re.compile(
        r"""(
            (?<=\n)\'{3}.*?\'{3}(?=\s*\n) |
            (?<=^)\'{3}.*?\'{3}(?=\s*\n) |
            (?<=\n)\'{3}.*?\'{3}(?=$) |
            (?<=^)\'{3}.*?\'{3}(?=$) |
            (?<=\n)\"{3}.*?\"{3}(?=\s*\n) |
            (?<=^)\"{3}.*?\"{3}(?=\s*\n) |
            (?<=\n)\"{3}.*?\"{3}(?=$) |
            (?<=^)\"{3}.*?\"{3}(?=$)
        )""",
        flags=re.DOTALL | re.VERBOSE | re.MULTILINE,
    )
    python_multi_line_count = re.findall(multi_line_python_regex, string)

    return python_multi_line_count


# Count the total number of line comments (inline + multi-line) in the given string
def extract_line_comments(text: str) -> int:
    multi_lines_lines = []
    multi_lines = multi_line_comments(text)
    for line in multi_lines:
        multi_lines_lines += line.split("\n")
    return len(multi_lines_lines) + count_inline_comment(text)


# Count the total number of words in line comments in the given string
def extract_comment_word_count(text: str) -> int:
    comment_word_count = 0
    comments = re.findall(r"(?<=\#).+", text)
    for comment in comments:
        comment_word_count += len(comment.split())
    return comment_word_count


# Comment Metrics
def extract_comment_metrics(code_df: pd.DataFrame) -> pd.DataFrame:
    code_df["LOCom"] = code_df["source"].progress_apply(lambda x: extract_line_comments(str(x)))
    code_df["CW"] = code_df["source"].progress_apply(lambda x: extract_comment_word_count(str(x)))
    return code_df


############## MARKDOWN METRICS ##############


# Count headers (H1, H2 , H3)
def extract_header1_count(text: str) -> int:
    count = 0
    if text[0:2] == "# ":
        count += 1
    for i in range(len(text) - 2):
        if text[i : i + 3] == " # " or text[i : i + 3] == "\n# ":
            count += 1
    return count


def extract_header2_count(text: str) -> int:
    count = 0
    if text[0:3] == "## ":
        count += 1
    for i in range(len(text) - 2):
        if text[i : i + 4] == " ## " or text[i : i + 4] == "\n## ":
            count += 1
    return count


def extract_header3_count(text: str) -> int:
    count = 0
    if text[0:4] == "### ":
        count += 1
    for i in range(len(text) - 3):
        if text[i : i + 5] == " ### " or text[i : i + 5] == "\n### ":
            count += 1
    return count


def extract_md_word_count(text: str) -> int:
    return len(text.split())


def extract_markdown_metrics(md_df: pd.DataFrame) -> pd.DataFrame:
    # Markdown Metrics
    md_df["LMC"] = md_df["source"].apply(lambda x: len(re.findall("\n", x)) + 1 if type(x) == str else 0)
    md_df["H1"] = md_df["source"].progress_apply(lambda x: extract_header1_count(str(x)))
    md_df["H2"] = md_df["source"].progress_apply(lambda x: extract_header2_count(str(x)))
    md_df["H3"] = md_df["source"].progress_apply(lambda x: extract_header3_count(str(x)))
    md_df["MW"] = md_df["source"].progress_apply(lambda x: extract_md_word_count(str(x)))
    columns_to_drop = ["source"]
    return md_df.drop(columns=columns_to_drop)


def cache_result(func):

    def wrapper(*args, **kwargs):
        args_hash = hashlib.blake2b(str(args).encode(), digest_size=10).hexdigest()
        filename = os.path.join(CACHE_PATH, f"{func.__name__}${args_hash}.json")

        if os.path.exists(filename):
            with open(filename, "r") as file:
                return json.load(file)

        result = func(*args, **kwargs)

        with open(filename, "w") as file:
            json.dump(result, file)

        return result

    return wrapper


@cache_result
def get_eap_score_dict(code_df_file_path: str, chunk_size: int) -> dict:
    chunk_reader = pd.read_csv(code_df_file_path, chunksize=chunk_size)
    api_list = []
    for i, chunk in enumerate(chunk_reader):
        chunk: pd.DataFrame
        logger.info(f"processing eap score dict: chunksize={chunk_size} chunk_index={i}")
        chunk.fillna("", inplace=True)
        api_list += get_list_of_all_api(code_df=chunk)
    return get_api_popularity_dict(api_list=api_list)
