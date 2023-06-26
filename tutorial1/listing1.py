import kfp
from kfp.components import create_component_from_func, OutputBinaryFile, InputBinaryFile 
from kfp import dsl

def load_data(output_path: OutputBinaryFile(str)):
    import pip
    pip.main(['install', 'scikit-learn', 'numpy'])
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import numpy as np

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    np.savez(output_path,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)  

def run_classifier(input_path: InputBinaryFile(str)):
    import pip
    pip.main(['install', 'scikit-learn', 'numpy'])
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    data=np.load(input_path)
    classifier = LogisticRegression(max_iter=1000)
    test_accuracy = classifier.fit(data['X_train'], data['y_train']).score(data['X_test'], data['y_test'])
    print(test_accuracy)

def iris_pipeline():
    BASE_IMAGE = 'mtr.devops.telekom.de/ai/python:latest'
    load_data_component = create_component_from_func(func=load_data, base_image=BASE_IMAGE)()
    run_classifier_component = create_component_from_func(func=run_classifier, base_image=BASE_IMAGE)(load_data_component.output)

kfp.compiler.Compiler().compile(iris_pipeline, 'listing1.yaml')
kfp.Client().create_run_from_pipeline_func(iris_pipeline, arguments={})