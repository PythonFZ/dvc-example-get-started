from src.prepare import Prepare
from src.featurization import Featurize
from src.train import Train
from src.evaluate import Evaluate
import zntrack
import znflow

if __name__ == "__main__":
    with zntrack.Project() as project:
        prepare = Prepare()
        with znflow.disable_graph():
            outs = prepare.outs
        featurize = Featurize(prepared=outs)
        with znflow.disable_graph():
            outs = featurize.features
        
        train = Train(features=outs)
        with znflow.disable_graph():
            model_pth = train.model

        eval = Evaluate(model=model_pth, features=outs)

    project.build()