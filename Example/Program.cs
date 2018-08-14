using System;
using System.Collections.Generic;
using System.Text;
using MatsurikaG;
using System.Linq;
using System.IO;
using System.ComponentModel;

namespace DAutoencoder_test
{
    class Program
    {
        static ArtificialNeuralNetwork Ann;
        static int framesize = 50;
        static void Main(string[] args)
        {

            double[][] Data;
            double[][] Rawdata;
            int DataSize = 0;
            int input = 0;
            int output = 0;
            List<double> delta = new List<double>();
            List<double> offset = new List<double>();
            List<int> peak = new List<int>();
            int pea = 327;

            //using (StreamReader SR = new StreamReader("Blue_train_input.csv"))
            using (StreamReader SR = new StreamReader("input.csv"))
            //using (StreamReader SR = new StreamReader("input.csv"))
            {
                //using (StreamReader SR2 = new StreamReader("Blue_train_output.csv"))
                using (StreamReader SR2 = new StreamReader("output.csv"))
                //using (StreamReader SR2 = new StreamReader("output.csv"))
                {
                    bool flag = false;
                    string line = SR.ReadLine();
                    string line2 = SR2.ReadLine();
                    List<double[]> _data = new List<double[]>();
                    List<double[]> _rawdata = new List<double[]>();
                    while (line != null)
                    {
                        string[] pline = line.Split(',');
                        string[] pline2 = line2.Split(',');
                        if (flag)
                        {

                            input = pline.Length;
                            output = pline2.Length;
                            flag = false;
                        }
                        else
                        {

                            double[] temp = new double[framesize * 2];
                            double[] inputdata = new double[framesize];
                            double[] outputdata = new double[framesize];


                            
                            for (int i = 0; i < framesize; i++)
                            {
                                inputdata[i] = Convert.ToDouble(pline[i]);
                                outputdata[i] = Convert.ToDouble(pline2[i]);
                            }
                            double max = inputdata.Max();
                            double min = Math.Min(inputdata.Min(), outputdata.Min());
                            pea = 0;
                            //pea = Array.IndexOf(inputdata, max);
                            //if (pea == -1)
                            //    pea = Array.IndexOf(outputdata, max);

                            //max = 1000;
                            min = 0;
                            _rawdata.Add(inputdata);

                            for (int i = 0; i < framesize; i++)
                            {
                                //int idx = pea - (int)(framesize / 2) + i;

                                temp[i] = (inputdata[i] - min) / (max - min);

                                //if (rnd.Next(1, 10) == 4)
                                //    temp[i] = 0;
                                

                                //if (inputdata[idx] > 300)
                                temp[i + framesize] = (inputdata[i] - outputdata[i]) / max;
                                //else
                                //    temp[i+ framesize] = 0;
                            }

                            delta.Add(max - min);
                            offset.Add(min);
                            peak.Add(pea);
                            _data.Add(temp);

                            DataSize++;
                        }
                        line = SR.ReadLine();
                        line2 = SR2.ReadLine();
                    }
                    Data = _data.ToArray();
                    Rawdata = _rawdata.ToArray();
                    SR.Close();
                    SR2.Close();
                    Console.WriteLine("Welcome to ANN .NET library. \nChoose a function next. " +
                        "\n1 = Train \n2 = Improve \n3 = Compute");

                    string index = Console.ReadLine();
                    switch (Convert.ToInt32(index))
                    {
                        case 1:
                            {
                                Train(Data, framesize, framesize);
                                break;
                            }
                        case 2:
                            {
                                Improve(Data, framesize, framesize);
                                break;
                            }
                        case 3:
                            {
                                Compute(Data, Rawdata, framesize, framesize, delta.ToArray(), offset.ToArray(), peak.ToArray());
                                break;
                            }
                    }
                }

            }
            Console.Read();
        }

        static private void Train(double[][] Data, int input, int output)
        {
            //set parameters in layers
            int numHidden = 100;
            int numHidden2 = 100;
            int maps = 16;
            int Deep1 = 20;
            int Deep2 = 40;
            int windows = 2;
            //create layers and construct them
            List<NNlayers> Nlist = new List<NNlayers>();
           
            NNlayers C1 = new NNlayers(NNlayers.Layers_family.Convolution, input, Deep1, 3, false);
            NNlayers C2 = new NNlayers(NNlayers.Layers_family.Maxpool, (input - 3 + 1), Deep1, 2);
            int numCov = (int)Math.Ceiling((double)(input - 3 + 1) / 2);
            NNlayers N1 = new NNlayers(NNlayers.Layers_family.BN, numCov * Deep1, numCov * Deep1);
            NNlayers H2 = new NNlayers(NNlayers.Layers_family.ReLU, numCov * Deep1, numCov * Deep1);
            
            NNlayers C3 = new NNlayers(NNlayers.Layers_family.Convolution, numCov, Deep2, 3, Deep1, false);
            NNlayers C4 = new NNlayers(NNlayers.Layers_family.Meanpool, (numCov - 3 + 1) , Deep2, 2);
            int numCov2 = (int)Math.Ceiling((double)(numCov - 3 + 1) / 2) * Deep2;
            //NNlayers N1 = new NNlayers(NNlayers.Layers_family.Affine, (int)Math.Ceiling((double)(input - mapsize + 1)/windows)* maps, numHidden);
            NNlayers N2 = new NNlayers(NNlayers.Layers_family.BN, numCov2, numCov2);
            //NNlayers N3 = new NNlayers(NNlayers.Layers_family.ReLU, numHidden, numHidden);

            NNlayers H3 = new NNlayers(NNlayers.Layers_family.ReLU, numCov2, numCov2);
            NNlayers N10 = new NNlayers(NNlayers.Layers_family.Affine, numCov2, output);
            //NNlayers N11 = new NNlayers(NNlayers.Layers_family.Sigmoid, output, output);
            Nlist.Add(C1);
            Nlist.Add(C2);
            Nlist.Add(N1);
            Nlist.Add(H2);
            Nlist.Add(C3);
            Nlist.Add(C4);
            //Nlist.Add(N1);
            Nlist.Add(N2);
            Nlist.Add(H3);
           
            Nlist.Add(N10);
            //Nlist.Add(N11);

            //create a NN class
            Ann = new ArtificialNeuralNetwork(Nlist.ToArray(), input, output);
            Ann.PositiveLimit = 0.7;//default = 0.7
            Ann.Batchsize = 100;
            int maxEpochs = 4000;
            double learnRate = 0.005;

            //create a error monitor backgroundworker
            BackgroundWorker BGW = new BackgroundWorker();
            BGW.DoWork += new DoWorkEventHandler(backgroundWorker_NN_DoWork);
            BGW.RunWorkerAsync(maxEpochs);

            //train
            Ann.TrainModel(Data, maxEpochs, learnRate, 0);



            double trainAcc = Ann.Accu_train;
            Console.Write("\nFinal accuracy on train data = " +
            trainAcc.ToString("F4"));

            double testAcc = Ann.Accu_test;
            Console.Write("\nFinal accuracy on test data = " +
            testAcc.ToString("F4"));
            Console.Write("\nTrain finish");

            string site = System.DateTime.Now.ToString("yyMMddHHmm") + "_learnproj";
            Directory.CreateDirectory(site);
            Ann.Save_network(site, learnRate);
            Ann.Save_H5files(site);

            Console.Write("\nVariables have been save in " + site);

        }

        static private void Improve(double[][] Data, int input, int output)
        {

            //create a NN class
            Ann = new ArtificialNeuralNetwork(input, output);
            Ann.PositiveLimit = 0.7;//default = 0.7
            int maxEpochs = 500;
            Ann.Batchsize = 30;
            double learnRate = 0.01;

            //import old learning project

            DirectoryInfo di = new DirectoryInfo(System.Environment.CurrentDirectory);

            int date = 0;
            string path = "";
            List<string> projects = new List<string>();

            Console.WriteLine("\nChoose a learning project:");
            int N = 0;
            foreach (var fi in di.GetDirectories())
            {
                if (fi.Name.Contains("_learnproj"))
                {
                    projects.Add(fi.Name);
                    Console.WriteLine(N + " = " + fi.Name);
                    N++;
                }
            }
            if (N == 0)
            {
                Console.Write("\nCannot Find any learnproject");
                return;
            }
            string n = Console.ReadLine();


            bool error = Ann.ImportOldProject(projects[Convert.ToInt32(n)]);
            if (!error)
            {
                Console.Write("\nCannot import learnproject");
                return;
            }

            //create a error monitor backgroundworker
            BackgroundWorker BGW = new BackgroundWorker();
            BGW.DoWork += new DoWorkEventHandler(backgroundWorker_NN_DoWork);
            BGW.RunWorkerAsync(maxEpochs);

            //train
            Ann.ImproveModel(Data, maxEpochs, learnRate, 0);

            double trainAcc = Ann.Accu_train;
            Console.Write("\nFinal accuracy on train data = " +
            trainAcc.ToString("F4"));

            double testAcc = Ann.Accu_test;
            Console.Write("\nFinal accuracy on test data = " +
            testAcc.ToString("F4"));
            Console.Write("\nTrain finish");

            string site = System.DateTime.Now.ToString("yyMMddHHmm") + "_learnproj";
            Directory.CreateDirectory(site);
            Ann.Save_network(site, learnRate);
            Ann.Save_H5files(site);

            Console.Write("\nVariables have been save in " + site);

        }

        static private void backgroundWorker_NN_DoWork(object sender, DoWorkEventArgs e)
        {
            int epoch = 0;

            int max = (int)e.Argument;
            while (epoch != max - 10)
            {
                if (Ann.trainerror != null)
                {
                    double a = Ann.trainerror[0];
                    double b = Ann.trainerror[1];
                    if (epoch != a)
                    {
                        epoch = (int)a;
                        Console.Write("\neapoch " + epoch + " = " + b.ToString("F4"));
                    }
                }
            }
        }

        static private void Compute(double[][] Data, double[][] Raw, int input, int output, double[] delta, double[] offset, int[] peak)
        {
            //create a NN class
            Ann = new ArtificialNeuralNetwork(input, output);
            Ann.PositiveLimit = 0.5;//default = 0.7
            //int maxEpochs = 100;
            //double learnRate = 0.05;

            //import old learning project

            DirectoryInfo di = new DirectoryInfo(System.Environment.CurrentDirectory);

            int date = 0;
            string path = "";
            List<string> projects = new List<string>();

            Console.WriteLine("\nChoose a learning project:");
            int N = 0;
            foreach (var fi in di.GetDirectories())
            {
                if (fi.Name.Contains("learnproj"))
                {
                    projects.Add(fi.Name);
                    Console.WriteLine(N + " = " + fi.Name);
                    N++;
                }
            }
            if (N == 0)
            {
                Console.Write("\nCannot Find any learnproject");
                return;
            }
            string n = Console.ReadLine();


            bool error = Ann.ImportOldProject(projects[Convert.ToInt32(n)]);
            if (!error)
            {
                Console.Write("\nCannot import learnproject");
                return;
            }
            
            string site = System.DateTime.Now.ToString("yyMMddHHmm") + "_Compute.csv";
            using (StreamWriter SW = new StreamWriter(site))
            {
                int i = 0;
                foreach (double[] temp in Data)
                {

                    double[] result = new double[framesize];
                    Array.Copy(temp, result, framesize);
                    //List<NNlayers> layers = Ann.
                    result = Ann.Compute(result);
                    string line = "";
                    int idx = peak[i];// - (int)(framesize/2);
                    foreach (double t in result)
                    {
                        Raw[i][idx] = (temp[idx] - t) * delta[i] + offset[i];
                        idx++;
                    }
                    foreach (double t in Raw[i])
                        line += t.ToString() + ",";
                    SW.WriteLine(line);
                    i++;
                }
                SW.Close();
            }

            Console.Write("\nResult have been save in " + site);
        }

    }
}
