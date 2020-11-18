from flask import Flask, jsonify, request, Response, make_response, send_file
import numpy as np
import pandas as pd
# from keras.models import load_model
# from keras.utils import np_utils
# from keras.models import Model,Sequential
# from sklearn.preprocessing import MinMaxScaler
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
# from keras.layers import BatchNormalization, Activation, Embedding
# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam
# from keras import backend as K

app = Flask(__name__)

@app.route('/lstm', methods=['POST'])
def get_data():
    form = request.get_json()
    dataArray = pd.DataFrame(form)
    print(dataArray.shape)
    # dataArray = data_preprocessing(dataArray)
    # x_test = np.expand_dims(dataArray, axis=2)
    # K.clear_session()
    # res = classify(x_test)
    # res2 = extract_feature(x_test).tolist()
    res = ['DK']
    res2 = [1.22,0.61,1.22,1.22 ,1.83,1.3]
    return jsonify(res,res2)
"""
def data_preprocessing(data):
    #print(data.shape)
    df = pd.read_csv('./Data File/train-去除基线.csv')
    df_T = df.T
    x = df_T.iloc[:,0:588]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(x)
    xx = scaler.transform(data)
    x_test = pd.DataFrame(xx)
    x_test.columns = x.columns
    #x_test.index = x.index
    return x_test

def classify(x_test):
    model = load_model("./Model/20190914111435_epoch_300_Loss_0.5410643815994263_acc_0.9230769276618958.h5")
    y_pred_prob = model.predict(x_test)
    y_pred = []
    for i in range(y_pred_prob.shape[0]):
        if y_pred_prob[i][0] > y_pred_prob[i][1]:
            y_pred.append('DK961')
        else:
            y_pred.append('Langdon')
    #print(y_pred)
    return y_pred

def extract_feature(x_test):
    model = load_model("./Model/20190914111435_epoch_300_Loss_0.5410643815994263_acc_0.9230769276618958.h5")
    FC2_layer_model = Model(inputs = model.input, outputs = model.get_layer(index=16).output)
    FC2_output = FC2_layer_model.predict(x_test)
    return FC2_output

def data_inverse_transform(data,variety):
    if (variety=='DK961'):
        # 真实DK
        df1 = pd.read_csv('./Data File/DK_real.csv')
        df1_T = df1.T
        real = df1_T.iloc[:,0:588]

        df2 = data
        df2_T = df2.T
        fake = df2_T.iloc[:,0:588]
        n = df2_T.shape[0]

        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(real)

        original_DK = scaler.inverse_transform(fake)
        original_DK = pd.DataFrame(original_DK)
        index = []
        for i in range(n):
            index.append('DK'+str(i+1))
        original_DK.index = index
        original_DK.T.to_csv('./Data File/DK_fake_original.csv',index=False)

    elif (variety=='Langdon'):
        # 真实LD
        df1 = pd.read_csv('./Data File/LD_real.csv')
        df1_T = df1.T
        real = df1_T.iloc[:,0:588]

        df2 = data
        df2_T = df2.T
        fake = df2_T.iloc[:,0:588]
        n = df2_T.shape[0]

        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(real)

        original_LD = scaler.inverse_transform(fake)
        original_LD = pd.DataFrame(original_LD)
        index = []
        for i in range(n):
            index.append('LD'+str(i+1))
        original_LD.index = index
        original_LD.T.to_csv('./Data File/LD_fake_original.csv',index=False)

@app.route('/generate', methods=['POST'])
def generate():
    form = request.get_json()
    number = form['g_num']
    variety = form['g_name']
    #print('>>>>>>>>><<<<<<<<<<<<<<')
    #print(number,variety)
    global DK_raw
    global LD_raw
    global DK_raw1
    global LD_raw1
    global DK_raw2
    global LD_raw2
    global DK_raw3
    global LD_raw3    
    DK_raw = pd.DataFrame()
    LD_raw = pd.DataFrame()    
    DK_raw1 = pd.DataFrame()
    LD_raw1 = pd.DataFrame()  
    DK_raw2 = pd.DataFrame()
    LD_raw2 = pd.DataFrame()  
    DK_raw3 = pd.DataFrame()
    LD_raw3 = pd.DataFrame()   
    K.clear_session()
    cgan = CGAN()
    cgan.test(gen_nums=number,variety=variety)

    if (variety == 'DK961'):
        fileName = 'DK_fake_original.csv'
        DK_raw = pd.concat([DK_raw1,DK_raw2,DK_raw3],axis=0)
        data_inverse_transform(DK_raw,variety)
    else:
        fileName = 'LD_fake_original.csv'
        LD_raw = pd.concat([LD_raw1,LD_raw2,LD_raw3],axis=0)
        data_inverse_transform(LD_raw,variety)

    res = make_response(send_file('./Data File/' + fileName, attachment_filename='generated data.csv', as_attachment=True))
    return res

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 196
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2     # 类别
        self.latent_dim = 100    # 隐空间维数

        optimizer = Adam(0.0001, 0.5)
        # Build and compile the discriminator
        
        self.discriminator = self.build_discriminator()   # 
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(128, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))         
        
        model.add(Dense(256))
        model.add(BatchNormalization(momentum=0.8))   
        
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))      
        
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)
        
    def build_discriminator(self):

        model = Sequential()
        
        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        
        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))     
        model.add(Dropout(0.4))
        
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        # modify
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])
        #print(type(model_input))

        validity = model(model_input)
        #print(type(validity))

        return Model([img, label], validity)
    
    def test(self, gen_nums=1, variety='DK961'):

        r, c = 1, 2      
        global DK_raw1
        global LD_raw1 
        global DK_raw2
        global LD_raw2
        global DK_raw3
        global LD_raw3 
        # Period 1
        self.generator.load_weights("Model/G_model_1_3997.hdf5")
        self.discriminator.load_weights("Model/D_model_1_3997.hdf5")

        for i in range(gen_nums):
            noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
            sampled_labels = np.arange(0, 2).reshape(-1, 1)

            gen_imgs = self.generator.predict([noise, sampled_labels])

            # 原始DK存csv
            if (variety=='DK961'):
                fake_sample0=gen_imgs[0,0,:,0]
                df = pd.DataFrame(fake_sample0)
                index = 'DK' + str(i)
                df.columns = [index]
                DK_raw1 = pd.concat([DK_raw1,df],axis=1)
            # 原始LD存csv 
            elif (variety=='Langdon'):           
                fake_sample1 = gen_imgs[1,0,:,0]
                df = pd.DataFrame(fake_sample1)
                index = 'LD' + str(i)
                df.columns = [index]
                LD_raw1 = pd.concat([LD_raw1,df],axis=1)
            else:
                pass        
            
        # Period 2
        self.generator.load_weights("Model/G_model_2_3997.hdf5")
        self.discriminator.load_weights("Model/D_model_2_3997.hdf5")

        for i in range(gen_nums):
            noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
            sampled_labels = np.arange(0, 2).reshape(-1, 1)

            gen_imgs = self.generator.predict([noise, sampled_labels])

            #global DK_raw2
            #global LD_raw2      

            # 原始DK存csv 
            if (variety=='DK961'): 
                fake_sample0=gen_imgs[0,0,:,0]
                df = pd.DataFrame(fake_sample0)
                index = 'DK' + str(i)
                df.columns = [index]
                DK_raw2 = pd.concat([DK_raw2,df],axis=1)
            # 原始LD存csv 
            elif (variety=='Langdon'):
                fake_sample1=gen_imgs[1,0,:,0]
                df = pd.DataFrame(fake_sample1)
                index = 'LD' + str(i)
                df.columns = [index]            
                LD_raw2 = pd.concat([LD_raw2,df],axis=1)
            else:
                pass

        # Period 3
        self.generator.load_weights("Model/G_model_3_3997.hdf5")
        self.discriminator.load_weights("Model/D_model_3_3997.hdf5")

        for i in range(gen_nums):
            noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
            sampled_labels = np.arange(0, 2).reshape(-1, 1)

            gen_imgs = self.generator.predict([noise, sampled_labels])

            # 原始DK存csv
            if (variety=='DK961'):
                fake_sample0=gen_imgs[0,0,:,0]
                df = pd.DataFrame(fake_sample0)
                index = 'DK' + str(i)
                df.columns = [index]
                df.loc[197] = 0
                DK_raw3 = pd.concat([DK_raw3,df],axis=1)
            elif (variety=='Langdon'):
            # 原始LD存CSV            
                fake_sample1=gen_imgs[1,0,:,0]
                df = pd.DataFrame(fake_sample1)
                index = 'LD' + str(i)
                df.columns = [index]          
                df.loc[197] = 1
                LD_raw3 = pd.concat([LD_raw3,df],axis=1)
            else:
                pass    
"""
if __name__ == '__main__':    
    app.run()