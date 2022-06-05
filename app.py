from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server

# import libraries
from prosemble import Hybrid
import numpy as np
import pickle

# 2. Create the app object
app = Flask(__name__)
pickle_in1 = open("svc.pkl", "rb")
pickle_in2 = open("knn.pkl", "rb")
pickle_in3 = open("dtc.pkl", "rb")

svc = pickle.load(pickle_in1)
knn = pickle.load(pickle_in2)
dtc = pickle.load(pickle_in3)


def get_posterior(x, y_, z_):
    """

    :param x: Input data
    :param y_: prediction
    :param z_: model
    :return: prediction probabilities
    """
    z1 = z_.predict_proba(x)
    certainties = [np.max(i) for i in z1]
    cert = np.array(certainties).flatten()
    cert = cert.reshape(len(cert), 1)
    y_ = y_.reshape(len(y_), 1)
    labels_with_certainty = np.concatenate((y_, cert), axis=1)
    return np.round(labels_with_certainty, 4)


# classes labels
proto_classes = np.array([0, 1])

# object of Hybrid class from prosemble
ensemble = Hybrid(model_prototypes=None, proto_classes=proto_classes, mm=2, omega_matrix=None, matrix='n')


def predict_BreastCancer():
    Radius_mean = input("Enter the Radius_mean：", type=FLOAT)
    Radius_texture = input("Enter the Radius_texture：", type=FLOAT)
    # Method = input("Enter the Method as soft or hard：", type=TEXT)
    Method = select('Method', ['soft', 'hard'])

    # prediction using the svc,knn and dtc models
    pred1 = svc.predict([[Radius_mean, Radius_texture]])
    pred2 = knn.predict([[Radius_mean, Radius_texture]])
    pred3 = dtc.predict([[Radius_mean, Radius_texture]])

    # confidence of prediction using the svc,knn and dtc models respectively
    sec1 = get_posterior(x=[[Radius_mean, Radius_texture]], y_=pred1, z_=svc)
    sec2 = get_posterior(x=[[Radius_mean, Radius_texture]], y_=pred2, z_=knn)
    sec3 = get_posterior(x=[[Radius_mean, Radius_texture]], y_=pred3, z_=dtc)
    all_pred = [pred1, pred2, pred3]
    all_sec = [sec1, sec2, sec3]
    # prediction from the ensemble using hard voting
    prediction1 = ensemble.pred_prob([[Radius_mean, Radius_texture]], all_pred)
    # prediction from the ensemble using soft voting
    prediction2 = ensemble.pred_sprob([[Radius_mean, Radius_texture]], all_sec)
    # confidence of the prediction using hard voting
    hard_prob = ensemble.prob([[Radius_mean, Radius_texture]], all_pred)
    # confidence of the prediction using soft voting
    soft_prob = ensemble.sprob([[Radius_mean, Radius_texture]], all_sec)
    if Method == 'soft':
        if prediction2[0] > 0.5:
            # put_text("WDBC-Benign {}".format(soft_prob[0]))
            put_text(f"Benign with {soft_prob[0] * 100}% confidence")
        else:
            # put_text("WDBC-Malignant w{}".format(soft_prob[0]))
            put_text(f"Malignant with {soft_prob[0] * 100}% confidence")

    if Method == 'hard':
        if prediction1[0] > 0.5:
            put_text(f"Benign with {hard_prob[0] * 100}% confidence")
        else:
            put_text(f"Malignant with {hard_prob[0] * 100}% confidence")


app.add_url_rule('/tool', 'webio_view', webio_view(predict_BreastCancer),
                 methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict_BreastCancer, port=args.port)

# if __name__ == '__main__':
# predict()

# app.run(host='localhost', port=80)

# visit http://localhost/tool to open the PyWebIO application.
