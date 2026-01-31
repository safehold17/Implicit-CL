import importlib.util
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXISTENCE_PATH = os.path.join(
    PROJECT_ROOT, "adapters", "ctrl_sim", "existence.py"
)
spec = importlib.util.spec_from_file_location("ctrlsim_existence", EXISTENCE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sim_position_exists = module.sim_position_exists


def test_sim_position_exists():
    assert sim_position_exists(0.0, 0.0) is True
    assert sim_position_exists(-10000.0, -10000.0) is False
    assert sim_position_exists(float("nan"), 0.0) is False


if __name__ == "__main__":
    test_sim_position_exists()
    print("OK")
