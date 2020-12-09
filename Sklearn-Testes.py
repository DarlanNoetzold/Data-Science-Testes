#!/usr/bin/env python
# coding: utf-8

# # Alucar - Analisando as vendas

# In[ ]:


import pandas as pd


# In[2]:


pd.read_csv('alucar.csv').head()


# In[ ]:


alucar = pd.read_csv('alucar.csv')


# In[4]:


print('Quantidade de linhas e colunas:', alucar.shape)


# In[5]:


print('Quantidade de dados nulos:', alucar.isna().sum().sum())


# In[6]:


alucar.dtypes


# In[7]:


alucar['mes'] = pd.to_datetime(alucar['mes'])
alucar.dtypes


# In[8]:


get_ipython().system('pip install seaborn==0.9.0')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[9]:


print(sns.__version__)


# In[10]:


sns.lineplot(x='mes', y='vendas', data=alucar)


# In[11]:


sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.lineplot(x='mes', y='vendas', data=alucar)
ax.figure.set_size_inches(12,6)
ax.set_title('Vendas Alucar de 2017 e 2018', loc='left', fontsize=18)
ax.set_xlabel('Tempo', fontsize=14)
ax.set_ylabel('Vendas (R$)', fontsize=14)
ax = ax


# In[12]:


alucar['aumento'] = alucar['vendas'].diff()
alucar.head()


# In[13]:


sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.lineplot(x='mes', y='aumento', data=alucar)
ax.figure.set_size_inches(12,6)
ax.set_title('Aumento das vendas da Alucar de 2017 e 2018', loc='left', fontsize=18)
ax.set_xlabel('Tempo', fontsize=14)
ax.set_ylabel('Aumento', fontsize=14)
ax = ax


# In[ ]:


def plotar(titulo, labelx, labely, x, y, dataset):
  sns.set_palette('Accent')
  sns.set_style('darkgrid')
  ax = sns.lineplot(x=x, y=y, data=dataset)
  ax.figure.set_size_inches(12,6)
  ax.set_title(titulo, loc='left', fontsize=18)
  ax.set_xlabel(labelx, fontsize=14)
  ax.set_ylabel(labely, fontsize=14)
  ax = ax


# In[15]:


plotar('Aumento das vendas da Alucar de 2017 e 2018', 'Tempo', 'Aumento',
      'mes', 'aumento', alucar)


# In[16]:


alucar['aceleracao'] = alucar['aumento'].diff()
alucar.head()


# In[17]:


plotar('Aceleração das vendas da Alucar de 2017 e 2018', 'Tempo', 'Aceleração',
      'mes', 'aceleracao', alucar)


# In[18]:


plt.figure(figsize=(16,12))
ax = plt.subplot(3,1,1)
ax.set_title('Análise de vendas da Alucar de 2017 e 2018', fontsize=18,loc='left')
sns.lineplot(x='mes', y='vendas', data=alucar)
plt.subplot(3,1,2)
sns.lineplot(x='mes', y='aumento', data=alucar)
plt.subplot(3,1,3)
sns.lineplot(x='mes', y='aceleracao', data=alucar)
ax = ax


# In[ ]:


def plot_comparacao(x, y1, y2, y3, dataset, titulo):
  plt.figure(figsize=(16,12))
  ax = plt.subplot(3,1,1)
  ax.set_title(titulo, fontsize=18,loc='left')
  sns.lineplot(x=x, y=y1, data=dataset)
  plt.subplot(3,1,2)
  sns.lineplot(x=x, y=y2, data=dataset)
  plt.subplot(3,1,3)
  sns.lineplot(x=x, y=y3, data=dataset)
  ax = ax


# In[20]:


plot_comparacao('mes', 'vendas', 'aumento', 'aceleracao',
               alucar, 'Análise das vendas da Alucar de 2017 e 2018')


# In[ ]:


from pandas.plotting import autocorrelation_plot


# In[22]:


ax = plt.figure(figsize=(12,6))
ax.suptitle('Correlação das vendas', fontsize=18, x=0.26, y=0.95)
autocorrelation_plot(alucar['vendas'])
ax = ax


# In[23]:


ax = plt.figure(figsize=(12,6))
ax.suptitle('Correlação do aumento', fontsize=18, x=0.26, y=0.95)
autocorrelation_plot(alucar['aumento'][1:])
ax = ax


# In[24]:


ax = plt.figure(figsize=(12,6))
ax.suptitle('Correlação da aceleração', fontsize=18, x=0.26, y=0.95)
autocorrelation_plot(alucar['aceleracao'][2:])
ax = ax


# # Alucar - Analisando assinantes da newsletter

# In[25]:


assinantes = pd.read_csv('newsletter_alucar.csv')
assinantes.head()


# In[26]:


assinantes.dtypes


# In[27]:


print('Quantidade de linhas e colunas:', assinantes.shape)
print('Quantidade de dados nulos:', assinantes.isna().sum().sum())


# In[28]:


assinantes['mes'] = pd.to_datetime(assinantes['mes'])
assinantes.dtypes


# In[29]:


assinantes['aumento'] = assinantes['assinantes'].diff()
assinantes['aceleracao'] = assinantes['aumento'].diff()
assinantes.head()


# In[30]:


plot_comparacao('mes', 'assinantes', 'aumento', 'aceleracao', 
                assinantes, 'Análise de assinantes da newsletter')


# # Chocolura - Analisando as vendas

# In[31]:


chocolura = pd.read_csv('chocolura.csv')
chocolura.head()


# In[32]:


chocolura.dtypes


# In[33]:


chocolura['mes'] = pd.to_datetime(chocolura['mes'])
chocolura.dtypes


# In[34]:


print('Quantidade de linhas:', chocolura.shape)
print('Quantidade de dados nulos:', chocolura.isna().sum().sum())


# In[35]:


chocolura['aumento'] = chocolura['vendas'].diff()
chocolura['aceleracao'] = chocolura['aumento'].diff()
chocolura.head()


# In[36]:


plot_comparacao('mes', 'vendas', 'aumento', 'aceleracao', 
                chocolura, 'Análise de vendas da Chocolura de 2017 a 2018')


# # Chocolura - Vendas diárias (Outubro e Novembro)

# In[37]:


vendas_por_dia = pd.read_csv('vendas_por_dia.csv')
vendas_por_dia.head()


# In[38]:


print('Quantidade de linhas e colunas:', vendas_por_dia.shape)
print('Quantidade de dados nulos:', vendas_por_dia.isna().sum().sum())


# In[39]:


vendas_por_dia.dtypes


# In[40]:


vendas_por_dia['dia'] = pd.to_datetime(vendas_por_dia['dia'])
vendas_por_dia.dtypes


# In[41]:


vendas_por_dia['aumento'] = vendas_por_dia['vendas'].diff()
vendas_por_dia['aceleracao'] = vendas_por_dia['aumento'].diff()
vendas_por_dia.head()


# In[42]:


plot_comparacao('dia', 'vendas', 'aumento', 'aceleracao',
               vendas_por_dia, 'Análise de vendas de Outubro e Novembro - Chocolura')


# **Analisando a sazonalidade**

# In[ ]:


vendas_por_dia['dia_da_semana'] = vendas_por_dia['dia'].dt.weekday_name


# In[44]:


vendas_por_dia.head(7)


# In[45]:


vendas_por_dia['dia_da_semana'].unique()


# In[ ]:


dias_traduzidos = {'Monday':'Segunda', 'Tuesday':'Terca', 'Wednesday':'Quarta',
                   'Thursday':'Quinta', 'Friday':'Sexta', 'Saturday':'Sabado',
       'Sunday':'Domingo'}


# In[47]:


vendas_por_dia['dia_da_semana'] = vendas_por_dia['dia_da_semana'].map(dias_traduzidos)
vendas_por_dia.head()


# In[48]:


vendas_por_dia.head(14)


# **Agrupando os dias**

# In[ ]:


vendas_agrupadas = vendas_por_dia.groupby('dia_da_semana')['vendas', 'aumento', 'aceleracao'].mean().round()


# In[50]:


vendas_agrupadas


# **Correlação das vendas diárias**

# In[51]:


ax = plt.figure(figsize=(12,6))
ax.suptitle('Correlação das vendas diárias', fontsize=18, x=0.3, y=0.95)
autocorrelation_plot(vendas_por_dia['vendas'])
ax = ax


# In[52]:


ax = plt.figure(figsize=(12,6))
ax.suptitle('Correlação do aumento das vendas diárias', fontsize=18, x=0.35, y=0.95)
autocorrelation_plot(vendas_por_dia['aumento'][1:])
ax = ax


# In[53]:


ax = plt.figure(figsize=(12,6))
ax.suptitle('Correlação da aceleração das vendas diárias', fontsize=18, x=0.35, y=0.95)
autocorrelation_plot(vendas_por_dia['aceleracao'][2:])
ax = ax


# # Cafelura - Análise de vendas

# In[54]:


cafelura = pd.read_csv('cafelura.csv')
cafelura.head()


# In[55]:


cafelura.dtypes


# In[56]:


cafelura['mes'] = pd.to_datetime(cafelura['mes'])
cafelura.dtypes


# In[57]:


print('Quantidade de linhas e colunas:', cafelura.shape)
print('Quantidade de dados nulos:', cafelura.isna().sum().sum())


# In[58]:


plotar('Vendas da Cafelura de 2017 e 2018', 'Tempo', 'Vendas',
      'mes', 'vendas', cafelura)


# In[59]:


quantidade_de_dias_de_fds = pd.read_csv('dias_final_de_semana.csv')
quantidade_de_dias_de_fds.head()


# In[60]:


quantidade_de_dias_de_fds['quantidade_de_dias'].values


# In[61]:


cafelura['vendas_normalizadas'] = cafelura['vendas'] / quantidade_de_dias_de_fds['quantidade_de_dias'].values
cafelura.head()


# In[62]:


plotar('Vendas normalizadas da Cafelura de 2017 a 2018',
      'Tempo', 'Vendas normalizadas', 'mes', 'vendas_normalizadas',
      cafelura)


# In[63]:


plt.figure(figsize=(12,8))
ax = plt.subplot(2,1,1)
ax.set_title('Vendas Cafelura 2017 e 2018', fontsize=18)
sns.lineplot(x='mes', y='vendas', data=cafelura)
ax = plt.subplot(2,1,2)
ax.set_title('Vendas Normalizadas Cafelura 2017 e 2018', fontsize=18)
sns.lineplot(x='mes', y='vendas_normalizadas', data=cafelura)
ax = ax


# # Statsmodels

# In[64]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[65]:


resultado = seasonal_decompose([chocolura['vendas']], freq=3,)
ax = resultado.plot()


# In[ ]:


observacao = resultado.observed
tendencia = resultado.trend
sazonalidade = resultado.seasonal
ruido = resultado.resid


# In[67]:


data = ({
    'observacao':observacao,
    'tendencia':tendencia,
    'sazonalidade':sazonalidade,
    'ruido':ruido
})

resultado = pd.DataFrame(data)
resultado.head()


# In[68]:


plot_comparacao(resultado.index, 'observacao', 'tendencia', 'sazonalidade', resultado,
               'Exemplo de Statsmodels')


# In[ ]:





# # Alucel - Análise de vendas

# In[70]:


alucel = pd.read_csv('alucel.csv')
alucel.head()


# In[71]:


alucel.dtypes


# In[73]:


alucel['dia'] = pd.to_datetime(alucel['dia'])
alucel.dtypes


# In[74]:


print('Quantidade de linhas e colunas:', alucel.shape)
print('Quantidade de dados nulos:', alucel.isna().sum().sum())


# In[75]:


alucel['aumento'] = alucel['vendas'].diff()
alucel['aceleracao'] = alucel['aumento'].diff()
alucel.head()


# In[76]:


plot_comparacao('dia', 'vendas', 'aumento', 'aceleracao',
                alucel, 'Análise de vendas da Alucel de Outubro e Novembro de 2018')


# **Média móvel**

# In[ ]:


alucel['media_movel'] = alucel['vendas'].rolling(7).mean()


# In[79]:


alucel.head(7)


# In[80]:


plotar('Análise de vendas com média móvel de 7 dias',
      'Tempo', 'Media móvel', 'dia', 'media_movel', alucel)


# In[ ]:


alucel['media_movel_21'] = alucel['vendas'].rolling(21).mean()


# In[82]:


plotar('Análise de vendas com média móvel de 21 dias',
      'Tempo', 'Media móvel', 'dia', 'media_movel_21', alucel)


# In[83]:


plot_comparacao('dia', 'vendas', 'media_movel', 'media_movel_21',
               alucel, 'Comparando as vendas com médias móveis')

