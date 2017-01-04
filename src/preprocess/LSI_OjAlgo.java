package preprocess;

import org.ojalgo.access.Access1D;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.ElementsConsumer;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;

import java.util.Iterator;
import java.util.Set;

/**
 * Created by nuplavikar on 12/29/16.
 */
/*
* As the name suggests this class uses the OjAlgo to compute LSI and other related
* management tasks such as storage etc.
* */
public class LSI_OjAlgo
{
    PrimitiveDenseStore C = null;
    int m;  //rowCount - Terms
    int n;  //columnCount - Documents
    boolean CFilled = false;
    boolean SVDComputed = false;
    boolean kRankApprox = false;


    MatrixStore<Double> U;
    MatrixStore<Double> V;
    MatrixStore<Double> Sigma;
    PrimitiveDenseStore Sigma_k;
    PrimitiveDenseStore U_k;
    PrimitiveDenseStore V_k;
    MatrixStore T;

    public LSI_OjAlgo(int m, int n)
    {
        //Create a Physical store factory. Using this factory we will allocate space to store the original
        //high-dimensional term-document matrix C
        final PhysicalStore.Factory<Double, PrimitiveDenseStore> doublePrimitiveDenseStoreFactory = PrimitiveDenseStore.FACTORY;
        //Actual allocation of matrix of size m x n, m: number of terms, n: number of documents
        C = doublePrimitiveDenseStoreFactory.makeZero(m, n);
        this.m = m;
        this.n = n;


        //TODO: Release unused rows from C after k-estimation
    }

    int populateTermDocMatrix(CollectionTFIDFVects collectionTFIDFVects, Set<String> setOfGlobalTerms)
    {
        if ( C == null )
        {
            System.err.println("Term-Document matrix C not allocated!");
            return -1;
        }

        Set<String> docFileNameSet = collectionTFIDFVects.getSetOfFileNames();
        Iterator<String> docFileNameIt = docFileNameSet.iterator();
        int docNo = 0;
        int termNo = 0;
        Double termWeight;
        String fileName, termStr;
        int numGlobTerms = setOfGlobalTerms.size();
        int numDocsColltn = collectionTFIDFVects.size();


        //Check if the number of terms is the same as the rows allocated
        if ( numGlobTerms != m )
        {
            System.err.println("ERROR!!! Inconsistency found. numGlobTerms:"+numGlobTerms+" and no. of Rows Allocated:"+m);
            return -1;
        }
        //Check if the number of documents is the same as the columns allocated
        if ( numDocsColltn != n )
        {
            System.err.println("ERROR!!! Inconsistency found. numDocsColltn:"+numDocsColltn+" and no. of Columns Allocated:"+n);
            return -2;
        }
        while ( docFileNameIt.hasNext() )
        {

            fileName = docFileNameIt.next();
            //Here order of keys and mappings should be the same as the  DocMagnitudeTreeMaps present in
            // collectionTFIDFVects are created according to GenerateTFIDFVector.globalTermIDFTreeMap. For more
            // information see GenerateTFIDFVector.getDocTFIDFVectors() method.
            DocMagnitudeTreeMap docVector = collectionTFIDFVects.getDocVector(fileName);
            Set<String> setOfDocTerms = docVector.getSetOfTerms();
            //To ensure consistency
            //if (setOfGlobalTerms.equals(setOfDocTerms)!=true)
            if ( setOfGlobalTerms.size() != setOfDocTerms.size() )
            {
                System.err.println("ERROR!! Terms and their order should be the same for both the doc. vectors and " +
                        "global idf vectors!!!");
                return -3;
            }
            Iterator<String> globalTermsIt = setOfGlobalTerms.iterator();
            termNo = 0;     //Reset the row counter
            while ( globalTermsIt.hasNext() )
            {
                termStr = globalTermsIt.next();
                termWeight = docVector.get(termStr);
                if ( termWeight == null )
                {
                    if ( docVector.containsKey(termStr) == false )
                    {
                        System.err.println("ERROR!!! Inconsistency found, term key:"+termStr+" NOT present in " +
                                "document vector for doc.:"+fileName);
                    }
                    else
                    {
                        System.err.println("ERROR!!! NULL value assigned as weight for term key:"+termStr+ " in " +
                                "document vector for doc.:"+fileName);
                    }
                    return -4;
                }
                C.set(termNo, docNo, termWeight);
                //increment the row number i.e. the term count
                termNo++;
            }

            //increment the column number i.e. the document count
            docNo++;
        }
        //Check if successfully filled-in the term-document matrix
        if ( (termNo!=m) || (docNo!=n) )
        {
            System.err.println("ERROR!!! While filling-in the term-document matrix C, <rows, cols>("+termNo+", "+docNo+
                    ") out of ("+m+", "+n+", "+docNo+") filled");
        }
        CFilled = true;
        return 0;
    }


    int printTermDocMatrix()
    {
        if (CFilled == false)
        {
            System.err.println("ERROR!!! Please fill the term-document matrix C");
            return -1;
        }
        System.out.println("Term-Document matrix: "+m+" x "+n);
        System.out.println(C);
        return 0;
    }

    /*
    * Computes the singular value decomposition. C should be filled as a pre-requisite
    * */
    int computeDecomposition()
    {

        if ( CFilled == false )
        {
            System.err.println("ERROR!!! Please fill the term-document matrix C");
            return -1;
        }
        final SingularValue<Double> svd = SingularValue.PRIMITIVE.make();
        System.out.println("\n\niscomputed = "+svd.isComputed());
        svd.decompose(C);
        System.out.println("\n\niscomputed = "+svd.isComputed());
        U = svd.getQ1();
        V = svd.getQ2();
        Sigma = svd.getD();

        System.out.println("\n\nU = "+U);
        System.out.println("\n\nSigma = "+Sigma);
        System.out.println("\n\nV = "+V);


        /*
        //Testing orthogonality
        System.out.println("\n\nU*Ut = "+U.multiply(U.transpose()));
        System.out.println("\n\nV*Vt = "+V.transpose().multiply(V));*/

        /*
        //Computing U*Sigma*(Vt)
        System.out.println("\n\n\nC = "+U.multiply(Sigma.multiply(V.transpose())));*/

        SVDComputed = true;
        return 0;
    }


    /*Computes the k-rank approximation of a matrix once, U, V and Sigma are computed*/
    int computeKRankApproximation(long k)
    {
        if ( CFilled == false )
        {
            System.err.println("ERROR!!! Please fill the term-document matrix C");
            return -1;
        }
        if ( SVDComputed == false )
        {
            System.err.println("ERROR!!! Please compute the SVD first");
            return -2;
        }
        if ( k >= Sigma.countRows() )
        {
            System.err.println("WARNING! current rank:"+Sigma.countRows()+" requested reduced rank k:"+k);
            return 0;
        }
        PhysicalStore SigmaApprox = Sigma.copy();

        //System.out.println("SigmaApprox:"+SigmaApprox);
        Sigma_k = truncateSigmaMemFriendly(k);
        U_k = truncateRightmostColumns((U.countColumns()-k), U);
        V_k = truncateRightmostColumns((V.countColumns()-k), V);

        //Destroy unnecessary objects
        Sigma = null;
        U = null;
        V = null;

        System.out.println("k = "+k);
        System.out.println("Sigma_k dimensions:"+Sigma_k.countRows()+" x "+Sigma_k.countColumns());
        System.out.println("U_k dimensions:"+U_k.countRows()+" x "+U_k.countColumns());
        System.out.println("V_k dimensions:"+V_k.countRows()+" x "+V_k.countColumns());
        //After truncation, sizes of the matrices are:
        //Sigma_k: k x k
        //U_k: m x k
        //V_k: n x k
        System.out.println("V_k:"+V_k);

        //Computing the scaled-up document matrix (V_k x Sigma_k)
        T = V_k.multiply(Sigma_k);
        //MatrixStore T = V_k;

        System.out.println("T:"+T);

        //Computing similarity
        int queryDocNum = 1;    //1 - second one
        Access1D<Double> query = T.sliceRow(queryDocNum, 0);
        System.out.println("queryDocNum:"+queryDocNum+"\t\tquery:"+query);

        long n = T.countRows(); //Store number of documents
        for ( int i=0; i<n; i++ )
        {
            Access1D<Double> candidateDoc = T.sliceRow(i, 0);
            double score = candidateDoc.dot(query);
            System.out.println("candidateDocNum:"+i+"\t\tcandidate:"+candidateDoc+"\t\tscore:"+score);
        }

        kRankApprox = true;
        return 0;
    }

    public PrimitiveDenseStore truncateSigmaMemFriendly(long k)
    {
        long rank;
        final PhysicalStore.Factory<Double, PrimitiveDenseStore> doublePrimitiveDenseStoreFactory = PrimitiveDenseStore.FACTORY;

//        PrimitiveArray primitiveArray = PrimitiveArray.make(k);
//        primitiveArray.fillMatching(Sigma.sliceDiagonal(0, 0));
//        System.out.println("primitiveArray: "+primitiveArray);
//        System.out.println(Sigma);
//
//        System.out.println("Sigma class "+Sigma.getClass());

        if ( Sigma.limitOfColumn(1000) != Sigma.limitOfRow(2000)  )
        {
            System.err.println("Sigma should be a square singular matrix: limitOfColumn:"+Sigma.limitOfColumn(1000)+" limitOfRow:"+Sigma.limitOfRow(2000));
        }
        System.out.println("Sigma--------->\n"+Sigma);
        System.out.println("Sigma: limitOfColumn:"+Sigma.limitOfColumn(1000)+" limitOfRow:"+Sigma.limitOfRow(2000));

        PrimitiveDenseStore Sigma_k = doublePrimitiveDenseStoreFactory.makeZero(k, k);
        for (int i =0; i<k; i++)
        {
            Sigma_k.set(i, i, Sigma.get(i, i));
        }

        //rank = Sigma.limitOfColumn()
        /*
        System.out.println("Checking if elements are pointed using the same references --v");
        Double a, b;
        for (int i =0; i<k; i++)
        {
            a = Sigma.get(i, i);
            b = Sigma_k.get(i, i);
            System.out.println("#"+i+"Sigma:"+a+" \tSigma_k:"+b);
            System.out.println("After new Double, a:"+a.hashCode()+" \tb:"+b.hashCode());
        }
        a = new Double(10);
        b = a;
        System.out.println("a:"+a.hashCode()+" \tb:"+b.hashCode());
        b = new Double(11);
        System.out.println("After new Double, a:"+a.hashCode()+" \tb:"+b.hashCode());
        Double c = Sigma.get(0, 0);
        c = c*10;
        Sigma_k.set(0, 0, c);
        for (int i =0; i<k; i++)
        {
            a = Sigma.get(i, i);
            b = Sigma_k.get(i, i);
            System.out.println("#"+i+"Sigma:"+a+" \tSigma_k:"+b);
            System.out.println("After new Double, a:"+a.hashCode()+" \tb:"+b.hashCode());
        }*/
        return Sigma_k;
    }

    public PrimitiveDenseStore truncateRightmostColumns(long truncateCount, MatrixStore<Double> matrixStore)
    {
        long rank, rowCnt, colCnt, newColCnt;
        final PhysicalStore.Factory<Double, PrimitiveDenseStore> doublePrimitiveDenseStoreFactory = PrimitiveDenseStore.FACTORY;

//        PrimitiveArray primitiveArray = PrimitiveArray.make(k);
//        primitiveArray.fillMatching(Sigma.sliceDiagonal(0, 0));
//        System.out.println("primitiveArray: "+primitiveArray);
//        System.out.println(Sigma);
//
//        System.out.println("Sigma class "+Sigma.getClass());

        if ( matrixStore.countColumns() < truncateCount  )
        {
            System.err.println("ERROR! Not sufficient columns to truncate. Existing column count:"+
                    matrixStore.countColumns()+" columnn truncation count:"+truncateCount);
            return null;
        }

        //DBG System.out.println("Matrix to be truncated--------->\n"+matrixStore);
        rowCnt = matrixStore.countRows();
        colCnt = matrixStore.countColumns();
        //DBG System.out.println("matrixStore: row:"+rowCnt+" columns:"+colCnt);
        newColCnt = (colCnt - truncateCount);
        PrimitiveDenseStore truncatedPrimitiveDenseStore = doublePrimitiveDenseStoreFactory.makeZero(rowCnt, newColCnt);
        for ( int i = 0; i<rowCnt; i++ )
        {
            for ( int j = 0; j < newColCnt; j++ )
            {
                truncatedPrimitiveDenseStore.set(i, j, matrixStore.get(i, j));
            }
        }
        //DBG System.out.println("Truncated matrix--------->\n"+truncatedPrimitiveDenseStore);
        //DBG rowCnt = truncatedPrimitiveDenseStore.countRows();
        //DBG colCnt = truncatedPrimitiveDenseStore.countColumns();
        //DBG System.out.println("Truncated prim. dense store: row:"+rowCnt+" columns:"+colCnt);

        return truncatedPrimitiveDenseStore;
    }
}
