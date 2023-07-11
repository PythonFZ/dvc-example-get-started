from src.prepare import Prepare
import zntrack

if __name__ == "__main__":
    with zntrack.Project() as project:
        prepare = Prepare()

    project.build()