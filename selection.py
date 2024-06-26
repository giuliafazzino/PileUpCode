"""
Select which data to keep based on the probability and reconstruct energy using them
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import ROOT
#import rootplotting as ap
#from root_numpy import fill_hist



def main():
    # Choose threshold to keep only signal
    threshold = .48

    # Read files
    data_path = 'data'
    data = pd.read_csv(data_path + '/df_test.csv') 

    # Re-order variables
    feature_names = ['clusterE', 'clusterEta',
                        'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                        'cluster_PTD', 'cluster_time', 'cluster_ISOLATION',
                        'cluster_SIGNIFICANCE', 'nPrimVtx']
    
    
    data_DNN = data[feature_names]


    
    # Predict response using network
    model = tf.keras.models.load_model(data_path + "/model_redRelu.h5", compile=False)
    if model == None:
        return
    data_DNN= data_DNN.assign(r_e_DNN = model.predict(data_DNN[feature_names].to_numpy()).flatten())
    
    ##############################################################
    #  Plotting reponse: for everything vs only selected events  #
    ##############################################################

    # Select events
    sig = (data['pred']) > threshold
    data_sel = data_DNN[sig]
    print(f'Total amount of events: {data.shape[0]}')
    print(f'Total amount of chosen events: {data_sel.shape[0]}')

    """
    # Plot
    try: 
        c = ROOT.TCanvas()
        xaxis = np.linspace(0, 10, 90 + 1, endpoint=True)
        #
        h_full = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        for i in  data_DNN['r_e_DNN'].to_numpy().flatten():
             h_full.Fill(i)
        h_full.Scale(1./h_full.Integral(), "width")
        h_full.SetLineColor(ROOT.kRed)

        h_sig = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        for j in data_sel['r_e_DNN'].to_numpy():
            h_sig.Fill(j)
        h_sig.Scale(1./h_sig.Integral(), "width")
        h_sig.SetLineColor(ROOT.kBlue)

        #c.Draw()
        h_full.Draw(option='HIST')
        h_sig.Draw(option='HIST SAME')
        ROOT.gPad.SetLogy(1)
        #l = ROOT.TLegend()
        #l.Add(h_full, 'full data')
        #l.Add(h_sig, 'chosen data')
        #l.Draw('SAME')
        #c.hist(h_sig, option='HIST', label = 'selected data', linecolor = ROOT.kGreen)
        #c.log()
        #c.xlabel('Response')
        #c.ylabel('Frequency')
        #c.text(["#sqrt{s} = 13 TeV", "p_{T} < SEE VALUE GeV", "threshold = 0.48"], qualifier = 'Simulation internal')
        #c.legend()
        #c.SaveAs('out/plots/responses.pdf')
    except AttributeError:
            print( 'out/plots/responses.pdf is not produced') """

    ##############################################################
    #  Plotting reponse mean and median vs threshold  #
    ##############################################################  


if __name__ == "__main__":
    main()