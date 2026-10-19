# Checkpoints de Modelos

Los modelos entrenados (`*.pt`, `*.pth`) se guardan automáticamente en este directorio durante la ejecución de los notebooks:
* `food101_resnet18_baseline.pt` (~43 MB)
* `food101_efficientnet_advanced.pt` (~17 MB)
* `ablation_frozen.pt` (~43 MB)
* `agnews_distilbert.pt` (~256 MB, modelo Transformer para AG News)
* `agnews_linearsvc_tfidf.joblib` (~1.7 MB, modelo clásico TF-IDF + LinearSVC para AG News)

> **Nota:** Los archivos de pesos binarios están excluidos del repositorio Git mediante `.gitignore` para cumplir con las políticas de límites de tamaño de archivo de GitHub (<100 MB). Se regeneran automáticamente al ejecutar las celdas de entrenamiento de los notebooks.
