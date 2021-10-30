import streamlit as slit 
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

def HidingStreamlitIcons():
    hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    slit.markdown(hide_streamlit_style, unsafe_allow_html=True)
def Header():
    slit.title('Applying Machine Learning for Intrusion detection Systems')
    slit.write()
    slit.markdown('----')
    slit.write()
    slit.markdown(':point_right:**Contributors: **')  
    cb_col1, cb_col2, cb_col3 = slit.columns(3)
    with cb_col1:
        slit.text('⇾ Venuri Kithmini')
    with cb_col2:
        slit.text('⇾ Malaka Fernando')
    with cb_col3:
        slit.text('⇾ Dakshitha Perera')

def AboutMachineLearning():
    slit.markdown(':point_right:** Training Accuracies of Machine Learning Algos **')   
    ml_col1, ml_col2, ml_col3 = slit.columns(3)
    with ml_col1:
        slit.markdown('**Random Forest: :point_down:**')
        slit.text('Accuracy - 0.9969')
        slit.text('precision - 0.9977')
        slit.text('recall - 0.9957')
        rf_cm_im = Image.open('Img/rf_img_cf.png')
        slit.image(rf_cm_im, use_column_width=True)

        slit.markdown('---')


        slit.markdown('**Naive Bayes: :point_down:**')
        slit.text('Accuracy - 0.9067')
        slit.text('precision - 0.9406')
        slit.text('recall - 0.8522')
        rf_cm_im = Image.open('Img/nb_img_cf.png')
        slit.image(rf_cm_im, use_column_width=True)

        
    with ml_col2:
        slit.markdown('**K-Nearest Neighbors: :point_down:**')
        slit.text('Accuracy - 0.9969')
        slit.text('precision - 0.9977')
        slit.text('recall - 0.9957')
        rf_cm_im = Image.open('Img/knn_img_cf.png')
        slit.image(rf_cm_im, use_column_width=True)

        slit.markdown('---')


        slit.markdown('**XG Boost: :point_down:**')
        slit.text('Accuracy - 0.9944')
        slit.text('precision - 0.9939')
        slit.text('recall - 00.9939')
        rf_cm_im = Image.open('Img/xgb_img_cf.png')
        slit.image(rf_cm_im, use_column_width=True)

    with ml_col3:
        slit.markdown('**Decision Tree: :point_down:**')
        slit.text('Accuracy - 0.9969')
        slit.text('precision - 0.9977')
        slit.text('recall - 0.9957')
        rf_cm_im = Image.open('Img/rf_img_cf.png')
        slit.image(rf_cm_im, use_column_width=True)

        slit.markdown('---')


        slit.markdown('**Ada Boost: :point_down:**')
        slit.text('Accuracy - 0.9886')
        slit.text('precision - 0.9891')
        slit.text('recall - 0.98627')
        rf_cm_im = Image.open('Img/ada_img_cf.png')
        slit.image(rf_cm_im, use_column_width=True)

class Predict:
    def upload_file(self):
        features = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
        'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'intrusion_type']
        uploaded_file = slit.file_uploader("Choose a file")
        if uploaded_file is not None:
            # , names=features, header=None
            df = pd.read_csv(uploaded_file, index_col=0)
            df_head = df.head()
            slit.markdown('---')
            slit.markdown('### Head of the dataset: :point_down:')
            slit.table(df_head)
            with slit.expander('Would you like to see the full datset 📚 '):
                slit.write(df)
            slit.markdown('---')
            with slit.expander('Expand this in order to predict the packet on differnt Machine Learning algorithms'):
                To_predict = df.drop('class', axis=1)
                self.Pre_To_Predict(To_predict)
                To_predict_df = self.Pre_To_Predict(To_predict)
                self.TestML(To_predict_df)

    def TestML(self, ToPrdictDF):
        try:
            rffilename = 'models/rf_model.sav'
            rf_model = pickle.load(open(rffilename, 'rb'))
            rf_says = rf_model.predict(ToPrdictDF)
        except:
            slit.error("ERROR: couldn\'t find the Random Forest model")
        try:
            nbfilename = 'models/nb_model.sav'
            nb_model = pickle.load(open(nbfilename, 'rb'))
            nb_says = nb_model.predict(ToPrdictDF)
        except:
            slit.error("ERROR: couldn\'t find the Naive bayes model")
        try:
            xgbfilename = 'models/xgb_model.sav'
            xgb_model = pickle.load(open(xgbfilename, 'rb'))
            xgb_says = xgb_model.predict(ToPrdictDF)
        except:
            slit.error("ERROR: couldn\'t find the XG Boost model")
        try:
            knnfilename = 'models/knn_model.sav'
            knn_model = pickle.load(open(knnfilename, 'rb'))
            knn_says = knn_model.predict(ToPrdictDF)
        except:
            slit.error("ERROR: couldn\'t find the KNN model")
        try:
            adafilename = 'models/Ada_model.sav'
            ada_model = pickle.load(open(adafilename, 'rb'))
            ada_says = ada_model.predict(ToPrdictDF)
        except:
            slit.error("ERROR: couldn\'t find the Ada Boost model")
        try:
            stackfilename = 'models/stack_model.sav'
            stack_model = pickle.load(open(stackfilename, 'rb'))
            stack_says = stack_model.predict(ToPrdictDF)
        except:
            slit.error("ERROR: couldn\'t find the Ada Boost model")
        try:
            slit.markdown("#### Machine Learning Results: :see_no_evil:")
            res_col1, res_col2 = slit.columns(2)
            slit.markdown('***')
            with res_col1:
                slit.text('k-nearest neighbor says: {}'.format(knn_says[0]))
                slit.text('Ada Boost says: {}'.format(ada_says[0]))
                slit.text('Support Vector Machine says: {}'.format(svm_says[0]))
            with res_col2:
                slit.text('Naive Bayes says: {}'.format(nb_says[0]))
                slit.text('Random Forest says: {}'.format(rf_says[0]))
                slit.text('XG Boost says: {}'.format(xgb_says[0]))
            slit.markdown('**Finally Ensemble Learning: :point_down:**')
            slit.text('Voting Classifier says: {}'.format(stack_says[0]))
            slit.text('Stacking Classifier says: {}'.format(voting_says[0]))
        except:
            slit.error(('Opps:exclamation: Something went wrong with Machine Learning Algos:weary:'))


    def Pre_To_Predict(self, To_predict):
        scaler = pickle.load(open('/home/nanoshadows/Downloads/scaler.pkl','rb'))
        single_col = To_predict.select_dtypes(include=['float64', 'int64']).columns
        To_predict1 = scaler.transform(To_predict.select_dtypes(include=['float64', 'int64'])) 
        To_predict1 = pd.DataFrame(To_predict1, columns = single_col)
        To_predict2_col = To_predict.select_dtypes(include=['object']).columns

        encoder_service = LabelEncoder()
        encoder_service.fit(['IRC', 'X11', 'Z39_50', 'auth', 'bgp', 'courier', 'csnet_ns',
                'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i',
                'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher',
                'hostnames', 'http', 'http_443', 'http_8001', 'imap5', 'iso_tsap',
                'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name',
                'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
                'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer',
                'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp',
                'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tim_i',
                'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois'])
        encoder_protocol_type = LabelEncoder()
        encoder_protocol_type.fit(['icmp', 'tcp', 'udp'])
        encoder_flag = LabelEncoder() 
        encoder_flag.fit(['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3',
            'SF', 'SH'])
        
        array_service = encoder_service.transform(To_predict['service'].values)
        array_protocol = encoder_protocol_type.transform(To_predict['protocol_type'].values)
        array_flag = encoder_flag.transform(To_predict['flag'].values)
        array_service, array_protocol, array_flag

        full_array = np.array([[array_protocol[0],array_service[0],array_flag[0]]])
        To_predict2 = pd.DataFrame(full_array, columns=To_predict2_col)
        Final_to_predict = pd.concat([To_predict1,To_predict2],axis=1)

        return Final_to_predict

def main():
    HidingStreamlitIcons()
    Header()
    slit.markdown('---')
    AboutMachineLearning()
    slit.markdown('---')
    slit.header('---Let\'s do a test, shall we:   :seedling:---')
    slit.write(' ')
    slit.markdown('#### Upload a csv file if you have one :point_down:')
    try:
        thePredict = Predict()
        thePredict.upload_file()
    except:
        slit.error('ERROR: Hmm Something went wrong')
    
main()
