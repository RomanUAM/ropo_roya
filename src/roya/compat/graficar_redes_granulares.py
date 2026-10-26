"""Mapas locales y errores; posiciones originales, sin mapa base inventado."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from redes_granulares_roya import OUT

def run():
    d=pd.read_csv(OUT/'comunidades_por_observacion.csv')
    d['periodo']=pd.to_datetime(d.Mes_registrado).dt.strftime('%Y-%m')+'-'+d.Quincena.str.extract(r'(\d+)')[0]
    # Todos los periodos para que el cambio temporal quede visible.
    for period,g in d.groupby('periodo',sort=True):
        g=g.drop_duplicates(['Coordenada_X','Coordenada_Y'])
        fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
        names=['topografia','simulada','condiciones15','ubicacion','urbanistica','fusion']
        for ax,name in zip(axes.flat,names):
            labels=pd.Categorical(g['comunidad_'+name]);colors=labels.codes
            cmap=plt.get_cmap('tab10',max(1,len(labels.categories)))
            ax.scatter(g.Coordenada_X/1000,g.Coordenada_Y/1000,c=colors,cmap=cmap,vmin=-.5,vmax=len(labels.categories)-.5,s=7,rasterized=True)
            handles=[Line2D([],[],marker='o',linestyle='',color=cmap(k),markersize=5,label='C'+str(category.rsplit('_',1)[-1])) for k,category in enumerate(labels.categories)]
            ax.legend(handles=handles,fontsize=7,loc='center right',framealpha=.9)
            ax.set_title(name+f' · {len(labels.categories)} categorías');ax.set_xlabel('X / km');ax.set_ylabel('Y / km');ax.set_aspect('equal',adjustable='box')
        fig.suptitle('Comunidades ambientales · '+period+'\nUbicaciones train + asignación inductiva de sitios reservados; EPSG:32614 provisional',fontsize=13)
        fig.savefig(OUT/('mapa_comunidades_'+period+'.png'),dpi=140);plt.close(fig)
    r=json.loads((OUT/'resultados.json').read_text());rows=[v for v in r['etapas'] if v['conjunto']=='prueba']
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,row in zip(axes,rows):
        a=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]])
        ax.imshow(a,cmap='Blues');ax.set_xticks([0,1],['Sin roya','Con roya']);ax.set_yticks([0,1],['Sin roya','Con roya']);ax.set_xlabel('Predicción');ax.set_ylabel('Registro observado');ax.set_title(row['etapa'])
        for (i,j),n in np.ndenumerate(a):ax.text(j,i,str(n),ha='center',va='center',color='white' if n>a.max()/2 else 'black',fontsize=13)
    fig.suptitle('Prueba exploratoria reservada por ubicación · errores de cada etapa')
    fig.savefig(OUT/'matrices_confusion.png',dpi=160);plt.close(fig)

if __name__=='__main__':run()
