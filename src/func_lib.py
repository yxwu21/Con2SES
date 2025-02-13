import re


def extract_epb_from_final_results(file_path):
    """
    Extracts the EPB value from the "FINAL RESULTS" section in a text file.

    :param file_path: Path to the text file.
    :return: The EPB value as a float, or None if not found or if the "FINAL RESULTS" section is not present.
    """

    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Regular expression to find the "FINAL RESULTS" section and then the EPB value
        final_results_pattern = r"FINAL RESULTS(.*?)\n\s*-+\n"
        epb_pattern = r"EPB\s*=\s*(-?\d+\.\d+)"

        final_results_match = re.search(final_results_pattern, content, re.DOTALL)

        if final_results_match:
            final_results_section = final_results_match.group(1)
            epb_match = re.search(epb_pattern, final_results_section)

            if epb_match:
                return float(epb_match.group(1))
            else:
                return None
        else:
            return None
    except Exception as e:
        return f"Error: {e}"

# file_path = "/home/junhal11/refined-mlses/datasets/benchmark_data_0.5_all/bench_full_0.55/case_1b/1bzk_A.ipb2.out.bench.full"
# epb_value = extract_epb_from_final_results(file_path)
# print(epb_value)