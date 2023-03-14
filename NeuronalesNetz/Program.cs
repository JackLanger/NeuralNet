// See https://aka.ms/new-console-template for more information

using MachineLearn.extension;
using NeuronalesNetz.algo;

NetworkModel nm = new NetworkModel(1,true);
nm.Train(
    gens: 5,
    gensize: 60000,
    learningrate: .12f,
    convolution: false,
    layers:30
    );

for (int i = 0; i < 10; i++)
{
    nm.Assess();
}

