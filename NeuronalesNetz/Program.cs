// See https://aka.ms/new-console-template for more information

using MachineLearn.extension;
using NeuronalesNetz.algo;

NetworkModel nm = new NetworkModel(1);
nm.Train(gens:8,gensize:5000);