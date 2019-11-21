def config_setting():
    setting = dict()
    setting['target_col'] = 'fraud_ind'
    setting['category_feat'] = {\
                'bacno','txkey','cano','contp',\
                'etymd','mchno','acqic','mcc',\
                'ecfg','insfg','stocn','scity',
                'stscd','ovrlt','flbmk','hcefg',\
                'csmcu','flg_3dsmk'}           
    return(setting)

if __name__ == "__main__":
    pass