from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")


y_pred = knn.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(confusion)

report = classification_report(y_test, y_pred)
print("Classification report:")
print(report)


param_grid = {'n_neighbors': [ 1, 3, 5, 7, 9]}

grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_
print(f"Best model: {best_model}")
print(f"Precision of the best model: {best_accuracy}")

