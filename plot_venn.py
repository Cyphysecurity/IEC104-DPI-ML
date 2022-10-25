import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
from pathlib import Path
import pandas as pd


def plot_venn():
    # First way to call the 2 group Venn diagram:
    plt.subplot(1, 2, 1)
    venn2(subsets = (130, 138, 100), set_labels = ('Number of Sessions in Y1', 'Number of Sessions in Y2'))
    plt.savefig('y1-y2-venn.eps')
    plt.show()
    
    # Second way
    #venn2([set(['A', 'B', 'C', 'D']), set(['D', 'E', 'F'])])
    #plt.show()

def plot_pie():
    # create data
    names='Station removed', 'Main server change', 'Single-server to dual-server', 'Dual-server to single-server',
    size=[23,1,3,2]
    
    # Create a circle for the center of the plot
    #plt.subplot(1, 2, 2)
    plt.figure(figsize=(15, 8))
    plt.rcParams['font.size'] = 18
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    plt.pie(size, labels=names, colors=['red','green','blue','skyblue'])
    p=plt.gcf()
    p.gca().add_artist(my_circle)

    plt.savefig('y1-y2-pie.eps')
    plt.show()

# Use Venn diagram for shared sessions in two years clustering used
# Use vertical barplot for session differences in two years
def plot_venn_hist(outpath):
    #fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 1, 1)
    v1 = venn2(subsets = (130, 138, 100), set_labels = ('Number of Sessions in Y1', 'Number of Sessions in Y2'))
    for t in v1.set_labels: t.set_fontsize(22)
    for t in v1.subset_labels: t.set_fontsize(20)
    # create data
    names=['Add stations', 'Remove stations', 'Single-server to dual-server', 'Dual-server to single-server']
    size=[30,-23,4,-4] # positive means adding from year 1 to year 2, negative means removal
    series1 = pd.Series(data=names, name='Reasons')
    series2 = pd.Series(data=size, name='Session_count_diff')
    df = pd.concat([series1, series2], axis=1)
    print(df.head())
    plt.subplot(2, 1, 2)
    # Create a circle for the center of the plot
    sns.set(style='whitegrid', font_scale=2)
    s1 = sns.barplot(x='Session_count_diff', y='Reasons', data=df, palette='Spectral')
    #s1.set_yticklabels(s1.get_yticks(), size = 2)
    
    #plt.rcParams['font.size'] = 18
    plt.savefig(Path(outpath, 'clustering-session-diff.pdf'))
    plt.show()

if __name__ == "__main__":
    print('Current path: ', Path().absolute())
    input_path = Path(Path().parent.parent.absolute(), 'input')
    output_path = Path(Path().parent.parent.absolute(), 'output')
    print('INPUT path: ', input_path)
    print('OUTPUT path: ', output_path)
    plot_venn_hist(output_path)
    #plot_venn()
    #plot_pie()
