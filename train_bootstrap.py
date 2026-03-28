import ast
import os
import sys


def get_parser():
    from arguments import parser

    return parser


def is_clearml_worker() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID"))


def _known_option_strings():
    option_strings = set()
    for action in get_parser()._actions:
        option_strings.update(action.option_strings)
    return option_strings


def cli_argv_contains_known_options(cli_argv) -> bool:
    known_options = _known_option_strings()
    return any(
        token == option or token.startswith(f"{option}=")
        for token in cli_argv
        for option in known_options
    )


def load_clearml_task_parameters():
    from clearml import Task

    task = Task.current_task()
    if task is None:
        task = Task.get_task(task_id=os.environ["CLEARML_TASK_ID"])
    return task.get_parameters() if task is not None else {}


def _parser_actions_by_dest():
    return {
        action.dest: action
        for action in get_parser()._actions
        if action.option_strings
    }


def _clearml_parameter_to_tokens(action, raw_value):
    if raw_value is None:
        return []

    # ClearML serializes unset optional scalars as empty strings.
    # Map those back to "missing" so argparse uses the parser default.
    if action.nargs in (None, "?") and raw_value == "" and action.default is None:
        return []

    if action.nargs in (None, "?"):
        return [str(raw_value)]

    if isinstance(raw_value, (list, tuple)):
        return [str(item) for item in raw_value]

    text = str(raw_value)
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]

    return text.split()


def clearml_parameters_to_argv(parameters):
    argv = []
    actions = _parser_actions_by_dest()

    for key, raw_value in parameters.items():
        dest = key.rsplit("/", 1)[-1]
        action = actions.get(dest)
        if action is None:
            continue

        option = action.option_strings[0]
        values = _clearml_parameter_to_tokens(action, raw_value)
        if not values:
            continue

        if action.nargs in (None, "?"):
            argv.append(f"{option}={values[0]}")
        else:
            argv.append(option)
            argv.extend(values)

    return argv


def resolve_runtime_args(cli_argv=None, clearml_parameters=None):
    parser = get_parser()
    cli_argv = list(sys.argv[1:] if cli_argv is None else cli_argv)

    if not is_clearml_worker():
        return parser.parse_args(cli_argv)

    if cli_argv_contains_known_options(cli_argv):
        return parser.parse_args(cli_argv)

    parameters = (
        clearml_parameters
        if clearml_parameters is not None
        else load_clearml_task_parameters()
    )
    fallback_argv = clearml_parameters_to_argv(parameters)
    return parser.parse_args(fallback_argv or cli_argv)


def setup_virtual_display():
    if not sys.platform.startswith("linux"):
        return None

    print("Setting up virtual display")
    import pyvirtualdisplay

    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
    display.start()
    return display
