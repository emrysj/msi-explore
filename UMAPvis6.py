#import tkinter as tk
#from tkinter import filedialog
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource,CustomJS, LassoSelectTool, BoxSelectTool, Slider, Select, RadioButtonGroup, Button, TextInput, HoverTool, TapTool
from bokeh.models.annotations import Title
from bokeh.plotting import figure, show, output_file, reset_output
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.server.server import Server
#from __future__ import division
from bokeh.io import curdoc   
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
import math
from operator import add

# This stops the GUI window opening  
root = tk.Tk()
root.withdraw()


#root.iconify()
#root.update()
#root.deiconify()
#root.mainloop()

# change data path to file path when ready

def load_data():
    curdoc().clear()
	
    button_1 = Button(label="Load data")
    button_1.on_click(load_data)
    button_2 = Button(label="Advanced options")
    button_2.on_click(Advanced_options)




    curdoc().add_root(button_1)
    curdoc().add_root(button_2)
	
    
    datapath=filedialog.askopenfilename() 
    

	
    global Dsource, Csource, Isource, Ssource, data_new_masked, data3, CX2, CY2, data, YY, XX,SelectedIon, mzlabs
    
    
    mz=pd.read_csv(datapath,sep='\t',skiprows=(0,1,2),header=None, nrows=1)
    
	
    mz=mz.drop(columns=[0,1,2])
    #lastmz=mz.columns[-1]
    #mz=mz.drop(columns=[lastmz-1,lastmz])

        
    data = pd.read_csv(datapath,sep='\t',skiprows=(0,1,2,3),header=None)
    Xpixels=data[1].tolist()
    Ypixels=data[2].tolist()

    last=data.columns[-1]
    data = data.drop(data.columns[[0, 1, 2,last-1,last]], axis=1)

    ScanNum=data.index.tolist()

    TotalScan=len(ScanNum)

    
    mzlabs=mz.loc[0].values.tolist()
    
    data.columns=mzlabs

    

    data = data.reindex(sorted(data.columns), axis=1)
    mzlabs.sort()

    peakNum=len(mzlabs)
    


# Work out pixel dimensions- need to do try/ catch here 

    a=Xpixels[1]
    Ypix=Xpixels.count(a)

    Xpix=np.round(TotalScan/Ypix)

    print(Ypix)
    print(Xpix)

# Make sure Ypix * Xpix = total pix 

# Do sum normalisation.. this will have multiple options 

# This is not the real sum normalisation 
    
    

  #  data_new = data.div(data.sum(axis=1), axis=0)

# This is a crude method for clearing background pixels based on lipid vs non-lipid 

    #low1=int(peakNum/2)
    #low2=int(peakNum/2)
    #high2=int(peakNum)
    
    low1=min(range(len(mzlabs)), key=lambda x:abs(mzlabs[x]-50))
    high1=min(range(len(mzlabs)), key=lambda x:abs(mzlabs[x]-200))
    low2=min(range(len(mzlabs)), key=lambda x:abs(mzlabs[x]-750))
    high2=min(range(len(mzlabs)), key=lambda x:abs(mzlabs[x]-900))
    
    
    D1=data.iloc[:,low1:high1]
    D2=data.iloc[:,low2:high2]
    D1s = D1.sum(axis=1)
    D1s=D1s+1
    D2s = D2.sum(axis=1)

    Ratio=D2s/D1s

    Ratio.tolist()
    
    del D1,D1s,D2,D2s

# This may be possible to do with only one copy of the data 


    data2=data
    data2.loc[Ratio<2,:]=0

    data3=data2.loc[~(data2==0).all(axis=1)]
    del data2
    data_new_masked = data3.div(data3.sum(axis=1), axis=0)
    data_new_masked=data_new_masked.fillna(0)

# Do PCA data reduction 
    
   



#Data_reduced=PCA(n_components=10).fit_transform(data_new)
    Data_reduced=PCA(n_components=10).fit_transform(data_new_masked)

# Perform the UMAP - these paramaters will be adjustable


    reducer = umap.UMAP(n_neighbors=10,min_dist=0.1,n_components=2,metric='euclidean')

    embedding = reducer.fit_transform(Data_reduced)

# This can be replaced using the Xpix Ypix from above 


    YY=int(Ypix)
    XX=int(Xpix)

    CX=[]
    for y in range(YY):
        for x in range(XX):
            CX.append(x)
        
    CY=[]

    for y in range(YY):
        for x in range(XX):
            CY.append(y)
		
		
		
    idx=data3.index

    CX2 = [CX[i] for i in idx]
    CY2 = [CY[i] for i in idx]
	
#    CX2=reverse(CX2)
#    CY2=reverse(CY2)

    #CX2=CX2[::-1]
    #CY2=CY2[::-1]
	
	
# This defines the UMAP output as columns for the plotting tools 

    x2=embedding[:, 0]
    y2=embedding[:, 1]    


    x3=x2-np.min(x2)
    y3=y2-np.min(y2)

    scannum= np.arange(0,TotalScan).tolist()
    spectra=scannum

    spectra2 = [spectra[i] for i in idx]

    ColX=(x3/np.max(x3))*255
    ColY=(y3/np.max(y3))*255
    CV1 = ["#%02x%02x%02x" % (int(r), int(g), 0) for r, g in zip(ColX, ColY)]
    CV2 = ["#%02x%02x%02x" % (0, int(r), int(g)) for r, g in zip(ColX, ColY)]
    CV3 = ["#%02x%02x%02x" % (int(r), 0, int(g)) for r, g in zip(ColX, ColY)]


# Create the data sources required 

    Mean1=np.mean(data3) #.iloc[1,:] 

    Blank=[0]*len(CX2)
    BlankMap = ["#%02x%02x%02x" % (0, 0, 0) for r in(ColX)]

    CompData=Mean1/Mean1
    Ssource = ColumnDataSource(data=dict(x=mzlabs,y=Mean1))
    Dsource = ColumnDataSource(data=dict(x=x2, y=y2, cordsX=CX2,cordsY=CY2,CV=CV1,spectra=spectra2))
    Csource = ColumnDataSource(data=dict(x=mzlabs,Region1=Mean1,Region2=Mean1,y=CompData))
    Isource = ColumnDataSource(data=dict(cordsX=CX2,cordsY=CY2,Region1=Blank,Region2=Blank,Map=Blank))



# Set up the plot region (need to define min and max for right plot) 
    TOOLS="lasso_select, box_select,pan,wheel_zoom,box_zoom,reset"


    Right=figure(title="UMAP output",plot_width=500,plot_height=500,x_range=[-15,15], y_range=[-15,15],tools=TOOLS)

    Left=figure(plot_width=500,plot_height=500,title=None,x_range=[0,XX], y_range=[0,YY],tools=TOOLS)
    Left.axis.visible = False
    Results=figure(plot_width=500,plot_height=400,title=None,x_range=[0,1200],tools="pan,wheel_zoom,box_zoom,reset,tap",x_axis_label='m/z',y_axis_label='log2 fold change')
    Spectrum=figure(plot_width=500,plot_height=400,title=None,x_range=[0,1200],x_axis_label='m/z',y_axis_label='mean intensity')
    SelectedIon=figure(plot_width=300,plot_height=300,title="Selected ion image",title_location = "below",x_range=[0,XX], y_range=[0,YY])
    SelectedIon.axis.visible = False
    Regions=figure(plot_width=200,plot_height=200,title=None,x_range=[0,Xpix], y_range=[0,Ypix],align="center")
    Regions.axis.visible = False


    Results.add_tools(HoverTool(
        tooltips = [
            ("m/z", "@x"),
            ("fold change", "@y"),
        ],
        mode='mouse',
        point_policy='snap_to_data'
    ))

# Populate the initial plots 
    r=Right.scatter(x='x',y='y',fill_color='CV',line_color=None,source=Dsource,radius=0.1)
    Left.square(x='cordsX',y='cordsY',fill_color='CV',line_color=None,alpha=1,size=5, source=Dsource)
#Spectrum.line(x='x',y='y',source=Ssource)
#Results.line(x='x',y='y',source=Csource)
    Spectrum.vbar(x='x',top='y',source=Ssource,width=0.5)
    Results.vbar(x='x',top='y',source=Csource,width=0.5)
    Regions.square(x='cordsX',y='cordsY',fill_color='Map',line_color=None,alpha=1,size=5, source=Isource)

    callback = CustomJS(args=dict(renderer=r), code="""
        renderer.glyph.radius = cb_obj.value;
    """)


    slider1 = Slider(start=0.01, end=1, step=0.01, value=0.1,title='circle size')
    slider1.js_on_change('value', callback)

    #text = TextInput(title="title", value='Insert experiment name')

    button_group = RadioButtonGroup(labels=["Red Green", "Red Blue", "Blue Green"], active=0)
    select = Select(title="Option:", value="foo", options=["Sum normalisation", "No normalisation", "Mean normalisation", "Median normalisation"])
    button_1 = Button(label="Load data")
    button_2 = Button(label="Reset data")
    button_3 = Button(label="Select region 1")
    button_4 = Button(label="Select region 2")
    button_5 = Button(label="Compare")
    button_6 = Button(label="Output report")
	# These are the list of actions possible 

    #text.on_change('value', update_title) 
    button_1.on_click(load_data)
    button_3.on_click(Region_1)
    button_4.on_click(Region_2)
    button_5.on_click(Compare_data)
    button_6.on_click(Output)
    Dsource.selected.on_change('indices', update_data)
#taptool = Results.select(type=TapTool)
    Csource.selected.on_change('indices',create_ion_map)


    
    p = gridplot([[Left,Right,column(slider1,select, button_3,button_4,Regions,button_5,button_6)],[Spectrum,Results,SelectedIon]])
    

    curdoc().add_root(p)
    #curdoc().title = "Data Plotter"

def Region_1():
    #Csource.data['Region1']=Mean2
    Region1=Dsource.selected.indices
    Mean2=np.mean(data_new_masked.iloc[Region1,:])
    Csource.data['Region1']=Mean2
    
    Mask1=[0]*len(CX2)
    for i in Region1:
        Mask1[i]=255
    Isource.data['Region1']=Mask1
    Mask2=Isource.data['Region2']
    Map = ["#%02x%02x%02x" % (int(r), 0, int(b)) for r, b in zip(Mask1, Mask2)]
    Isource.data['Map']=Map
    

def Region_2():
    Region2=Dsource.selected.indices
    Mean3=np.mean(data_new_masked.iloc[Region2,:])
    Csource.data['Region2']=Mean3
    #
    Mask2=[0]*len(CX2)
    
    for i in Region2:
        Mask2[i]=255
    Isource.data['Region2']=Mask2
    Mask1=Isource.data['Region1']
    Map = ["#%02x%02x%02x" % (int(r), 0, int(b)) for r, b in zip(Mask1, Mask2)]
    Isource.data['Map']=Map


	
def Compare_data():
    # Open a new tab and show results
    Mean2=Csource.data['Region1']
    Mean3=Csource.data['Region2']
    Compare=np.log2(Mean2/Mean3)
    Csource.data['y']=Compare
    
    

def update_title(attrname, old, new):
    Right.title.text = text.value
    
   

    
def update_data(attrname, old, new):
   
    scan=Dsource.selected.indices
    Mean1=np.mean(data3.iloc[scan,:])
    Ssource.data['y']=Mean1
	
def create_ion_map(attrname, old, new):
    mass=Csource.selected.indices[0] # Need to limit this to only one value 
    X=data.iloc[:,mass].values.tolist()
    X[X == math.inf] = 0
    X2=np.reshape(X,[YY,XX])
    X2=np.flipud(X2)
    SelectedIon.image(image=[X2], x=0, y=0, dw=XX, dh=YY, palette="Spectral11")
    ma1=mzlabs[mass]
    ma2=str(ma1)
    SelectedIon.title.text='m/z ' + ma2
	
    #SelectedIon.title.text=data.columns[mass]
    SelectedIon.title.align = "center"
    #SelectedIon.title_location = "below"

	
	
def main(curdoc):

    curdoc.add_root(button_1)
    curdoc.add_root(button_2)
	#doc.title = "MSI explore"


#curdoc().add_root(button_1)
	
	#doc.add_root(row(inputs, plot, width=800))
    
#button_1 = Button(label="Load data")
#button_1.on_click(load_data)



#curdoc().add_root(button_1)

def Output():
    Compare=Csource.data['y']
    Compare2=pd.DataFrame(mzlabs,Compare)
    Compare2=Compare2.sort_index(axis=0, level=None, ascending=False) 
    pp=len(Compare2)
    Compare3=Compare2.iloc[0:50, :]
    Compare4=Compare2.iloc[pp-50:pp,:]
    Compare3.to_csv('20201103_red.csv',sep=",",header=False)
    Compare4.to_csv('20201103_blue.csv',sep=",",header=False)

	
	
	
def Advanced_options():
    print('There will be things here')


button_1 = Button(label="Load data")
button_1.on_click(load_data)

button_2 = Button(label="Advanced options")
button_2.on_click(Advanced_options)

app = Application(FunctionHandler(main))
server = Server({'/': app}, port=0)

server.start()
server.show('/')

# Outside the notebook ioloop needs to be started
from tornado.ioloop import IOLoop
loop = IOLoop.current()
loop.start()
