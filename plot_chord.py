'''
https://coderzcolumn.com/tutorials/data-science/how-to-plot-chord-diagram-in-python-holoviews
https://holoviews.org/gallery/demos/bokeh/route_chord.html
'''

import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.airport_routes import routes, airports
from bokeh.sampledata.les_mis import data
from bokeh.plotting import show
from pathlib import Path, WindowsPath
import csv, re
hv.extension('bokeh')
br = hv.renderer('bokeh')
hv.output(size=500)
#hv.extension('matplotlib')
#hv.output(fig='svg', size=500)

# web browser and driver option
from selenium.webdriver import Chrome, ChromeOptions
options = ChromeOptions()
options.add_argument('--headless')
web_driver = Chrome(executable_path=r'C:\Windows\chromedriver.exe', options=options)



def plot_chord(infile):
    df = pd.read_csv(infile, delimiter=',')
    #df = df[df['clusters'] == 2]
    print(df.columns)
    packetCnt = pd.concat([df.rtu, df.server, df.numOfPackets], axis=1).sort_values('numOfPackets')
    node_df = pd.DataFrame(packetCnt.rtu.unique().tolist() + packetCnt.server.unique().tolist(), columns=['states'])
    nodes = hv.Dataset(node_df, 'states')
    chord_plot = hv.Chord((packetCnt, nodes), ['rtu', 'server'], ['numOfPackets'])
    chord_plot.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('rtu').str(),
                   labels='states', node_color=dim('states').str()
                  # width=500, height=500
                   ),
        opts.Labels(text_font_size='10pt'))
    #show(hv.render(chord_plot))

    # try exportation with Bokeh backend to svg
    from bokeh.io import export_svg
    p = hv.render(chord_plot, backend='bokeh')
    p.output_backend = "svg"
    #export_svg(p, filename="chord.svg", webdriver=web_driver, width=500, height=500)
    export_svg(p, filename="chord.svg", webdriver=web_driver)

    # render chord diagram as SVG figure with Matplotlib
    #hv.save(chord_plot, 'chord.svg', backend='matplotlib')
    #hv.render(chord_plot)

def example2():
    # Count the routes between Airports
    route_counts = routes.groupby(['SourceID', 'DestinationID']).Stops.count().reset_index()
    nodes = hv.Dataset(airports, 'AirportID', 'City')
    chord = hv.Chord((route_counts, nodes), ['SourceID', 'DestinationID'], ['Stops'])

    # Select the 20 busiest airports
    busiest = list(routes.groupby('SourceID').count().sort_values('Stops').iloc[-20:].index.values)
    busiest_airports = chord.select(AirportID=busiest, selection_mode='nodes')
    busiest_airports.opts(
        opts.Chord(cmap='Category20', edge_color=dim('SourceID').str(),
                   height=800, labels='City', node_color=dim('AirportID').str(), width=800))
    show(hv.render(busiest_airports))

def example1():
    links = pd.DataFrame(data['links'])
    print(links.columns, len(data))

    print(links.head(3))

    #show(hv.render(hv.Chord(links)))

    nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
    nodes.data.head()

    chord = hv.Chord((links, nodes)).select(value=(5, None))
    chord.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(),
                   labels='name', node_color=dim('index').str()))

    show(hv.render(chord))



if __name__ == "__main__":
    cur_p = Path()
    home_p = cur_p.home()
    in_p = Path(cur_p.absolute(), 'input')
    out_p = Path(cur_p.absolute(), 'output')
    print(
        'current directory: {} \nhome directory: {}\noutput: {} \n'.format(cur_p.absolute(), home_p, out_p.absolute()))

    #in_p = Path(cur_p.absolute().parents[2], 'Google Drive', 'IEC104', 'Netherlands-results', 'Clustering')
    print('input directory: ', in_p)
    csvfn = 'encoded_104JavaParser_2021-03-17_01_45_10_104only-0706-to-0710.csv'
    plot_chord(Path(in_p, 'clustering-gas', csvfn))
    #example2()
    #example1()

# %%


