import pandas as pd
import keras
from keras.models import Model, load_model

from data import prepare_data_test

def main():
    test_img_1, test_img_2, angles, ids = prepare_data_test()
    model = load_model('net0')
    out = model.predict(x=[test_img_1, test_img_2, angles]).flatten()
    d = {'id' : ids, 'is_iceberg' : out}

    pd.DataFrame(data=d).to_csv('out.csv', index=False)   
    print('Done')
if __name__ == '__main__':
    main()
