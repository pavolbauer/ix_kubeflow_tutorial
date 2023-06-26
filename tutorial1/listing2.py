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
    # or via mounted PVC
    np.savez('/home/src/iris.npz',X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)  

def run_classifier(input_path: InputBinaryFile(str)):
    import pip
    pip.main(['install', 'scikit-learn', 'numpy'])
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    data=np.load(input_path)
    # or via mounted PVC
    data=np.load('/home/src/iris.npz')
    classifier = LogisticRegression(max_iter=1000)
    test_accuracy = classifier.fit(data['X_train'], data['y_train']).score(data['X_test'], data['y_test'])
    print(test_accuracy)

def iris_pipeline():
    BASE_IMAGE = 'mtr.devops.telekom.de/ai/python:latest'
    
    resource_component = kfp.dsl.VolumeOp(
    name = "create-pvc",
    resource_name = 'name',
    modes = kfp.dsl.VOLUME_MODE_RWO,
    size = '10Gi',
    storage_class = 'standard')
    pipeline_volume = resource_component.volume
    # this creates a new volume, alternatively you can use an existing volume using dsl.PipelineVolume(pvc="existing-volume-name")
    
    load_data_component = create_component_from_func(func=load_data, base_image=BASE_IMAGE)()
    load_data_component.add_pvolumes({"/home/src":pipeline_volume})
    run_classifier_component = create_component_from_func(func=run_classifier, base_image=BASE_IMAGE)(load_data_component.output)
    run_classifier_component.add_pvolumes({"/home/src":pipeline_volume})

kfp.compiler.Compiler().compile(iris_pipeline, 'listing2.yaml')
kfp.Client().create_run_from_pipeline_func(iris_pipeline, arguments={})