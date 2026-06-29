# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:26:19 2023

@author: JUANMURILLOMORERA
"""
#Ejercicio #1

#Resuelto manualmente

#Ejercicio #2
#2.1
class MatrizDeConfusion:

  def __init__(self, p_mc): # Constructor
    #Atributos privados radio
    self.__mc = p_mc
  
  #GETS
  @property
  def mc(self):
    return self.__m
  
  #SETS
  @mc.setter
  def mc(self, p_mc):
      self.__mc = p_mc
 #
  def calcularMatrizDeConfusion(self):
    if (len(self.__mc) == 2 and len(self.__mc[0])==2):
      VP, VN, FP, FN = self.__mc[0][0],self.__mc[1][1],self.__mc[0][1],self.__mc[1][0]
      PG = (VP + VN)/(VP + VN + FP + FN)
      EG = 1-PG
      PP = VP/(VP + FP)
      PN = VN/(VN + FN)
      PFP = FP/(VN + FP)
      PFN = FN/(VP + FN) 
      AP =  VP/(VP + FN)
      AN =  VN/(VN + FP)
      diccionarioMC = {"Precision Global": PG, 
                       "Error Global": EG,
                       "Presicion Positiva (PP)": PP,
                       "Presicion Negativa (PN)": PN,
                       "Proporcion de Falsos Positivos (PFP)": PFP,
                       "Proporcion de Falsos Negativos (PFN)": PFN,
                       "Asertividad Positiva (AP)": AP,
                       "Asertividad Negativa (AN)": AN}
                       
      return diccionarioMC
    else:
      return "La matriz no es de 2X2"       
        
  def __str__(self):
     return f'Objeto Matriz Confusion: mc: {self.__mc}'
    
MC = [[892254,252],
      [9993,270]]
Mc = MatrizDeConfusion(MC)
print(Mc.calcularMatrizDeConfusion())

#2.2 Es bueno o malo el modelo predictivo
