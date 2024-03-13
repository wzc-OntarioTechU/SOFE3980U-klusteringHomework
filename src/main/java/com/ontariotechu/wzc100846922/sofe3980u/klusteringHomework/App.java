package com.ontariotechu.wzc100846922.sofe3980u.klusteringHomework;

import java.util.Iterator;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

import net.sf.javaml.clustering.DensityBasedSpatialClustering;
import net.sf.javaml.clustering.FarthestFirst;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.BICScore;
import net.sf.javaml.clustering.evaluation.Gamma;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import net.sf.javaml.distance.EuclideanDistance;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        DefaultDataset dataset = new DefaultDataset();
        
        // read in data set and insert into DefaultDataset object
        try {
			Scanner scan = new Scanner(new File("./src/main/resources/iris.data"));
			while (scan.hasNextLine()) {
				String[] tokens = scan.nextLine().split(",");
				double[] dataVector = new double[4];
				for (int i = 0; i < dataVector.length; i++) {
					dataVector[i] = Double.parseDouble(tokens[i]);
				}
				SparseInstance insInstance = new SparseInstance(dataVector, tokens[4]);
				dataset.add(insInstance);
			}
			scan.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(1);
		}
        
        // print the dataset to the console
        System.out.println("Dataset input:");
        printDataset(dataset);
        System.out.print("\n\n\n");
        
        // k clustering
        long kTimeStart, kTimeEnd;
        KMeans kMeansClusterer = new KMeans(3);
        Dataset[] kClusters;
        kTimeStart = System.nanoTime();
        kClusters = kMeansClusterer.cluster(dataset);
        kTimeEnd = System.nanoTime();
        
        // farthest first using Euclidean Distance measure
        long fTimeStart, fTimeEnd;
        FarthestFirst farthestFirstClusterer = new FarthestFirst(3, new EuclideanDistance());
        Dataset[] fClusters;
        fTimeStart = System.nanoTime();
        fClusters = farthestFirstClusterer.cluster(dataset);
        fTimeEnd = System.nanoTime();
        
        // density based spatial clustering using defaults of epsilon 0.1 minPoints 6 and Normalized Euclidean Distance measure
        long dTimeStart, dTimeEnd;
        DensityBasedSpatialClustering densityClusterer = new DensityBasedSpatialClustering();
        Dataset[] dClusters;
        dTimeStart = System.nanoTime();
        dClusters = densityClusterer.cluster(dataset);
        dTimeEnd = System.nanoTime();
        
        // Initialize common evaluators
        AICScore aicScorer = new AICScore(3);
        BICScore bicScorer = new BICScore(3);
        SumOfSquaredErrors sumSquareScorer = new SumOfSquaredErrors();
        Gamma gammaScorer = new Gamma(new EuclideanDistance()); // gamma using euclidean distance
        
        // Evaluate k Clustering;
        System.out.println("K-Clustering Cluster Datasets:");
        for (int i = 0; i < kClusters.length; i++) {
        	System.out.println("Cluster Dataset " + (i + 1) + ":");
        	printDataset(kClusters[i]);
        	System.out.println();
        }
        System.out.println("K-Cluster running time: " + (kTimeEnd - kTimeStart) + "ns");
        System.out.println("K-Cluster AIC Score: " + aicScorer.score(kClusters));
        System.out.println("K-Cluster BIC Score: " + bicScorer.score(kClusters));
        System.out.println("K-Cluster Sum of Squared Errors Score: " + sumSquareScorer.score(kClusters));
        System.out.println("K-Cluster Euclidean Gamma Score: " + gammaScorer.score(kClusters));
        
        // Evaluate f Clustering;
        System.out.println("Farthest First Clustering Cluster Datasets:");
        for (int i = 0; i < fClusters.length; i++) {
        	System.out.println("Cluster Dataset " + (i + 1) + ":");
        	printDataset(fClusters[i]);
        	System.out.println();
        }
        System.out.println("F-Cluster running time: " + (fTimeEnd - fTimeStart) + "ns");
        System.out.println("F-Cluster AIC Score: " + aicScorer.score(fClusters));
        System.out.println("F-Cluster BIC Score: " + bicScorer.score(fClusters));
        System.out.println("F-Cluster Sum of Squared Errors Score: " + sumSquareScorer.score(fClusters));
        System.out.println("F-Cluster Euclidean Gamma Score: " + gammaScorer.score(fClusters));
        
     // Evaluate f Clustering;
        System.out.println("Density Based Spatial Clustering Cluster Datasets:");
        for (int i = 0; i < dClusters.length; i++) {
        	System.out.println("Cluster Dataset " + (i + 1) + ":");
        	printDataset(dClusters[i]);
        	System.out.println();
        }
        System.out.println("D-Cluster running time: " + (dTimeEnd - dTimeStart) + "ns");
        System.out.println("D-Cluster AIC Score: " + aicScorer.score(dClusters));
        System.out.println("D-Cluster BIC Score: " + bicScorer.score(dClusters));
        System.out.println("D-Cluster Sum of Squared Errors Score: " + sumSquareScorer.score(dClusters));
        System.out.println("D-Cluster Euclidean Gamma Score: " + gammaScorer.score(dClusters));
    }
    
    /**
     * Used to print out a Dataset of iris
     * 
     * @param set The Dataset to print
     */
    private static void printDataset(Dataset set) {
    	Iterator<Instance> instIter = set.iterator();
        while(instIter.hasNext()) {
        	Instance rdInstance = instIter.next();
        	System.out.println("[ " + rdInstance.get(0) + " , " + rdInstance.get(1) + " , " + rdInstance.get(2) + " , " + rdInstance.get(3) + " ] " + rdInstance.classValue().toString());
        }
    }
}
