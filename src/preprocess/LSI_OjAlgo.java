package preprocess;

import org.ojalgo.access.Access1D;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.ElementsConsumer;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
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
    long m;  //rowCount - Terms
    long n;  //columnCount - Documents
    long k;  //Reduced m after using SVD and k-approximation; only used in client context

    boolean clientContext = false;
    boolean forBinaryVector = false;

    boolean CFilled = false;
    boolean SVDComputed = false;
    boolean kRankApprox = false;
    boolean scaledupTermVectComputed = false;
    PhysicalStore.Factory<Double, PrimitiveDenseStore> doublePrimitiveDenseStoreFactory;


    MatrixStore<Double> U;
    MatrixStore<Double> V;
    MatrixStore<Double> Sigma;
    PrimitiveDenseStore Sigma_k;
    PrimitiveDenseStore U_k;
    PrimitiveDenseStore V_k;
    MatrixStore<Double> T;//For server: stores the k approx. scaled up doc. matrix[n x k]. For client: stores the k approx.
    // scaled up query(doc.) matrix[1 x k]

    double max, min, negativeCorrection;
    boolean negativeCorrectionAdded=false;

    //constant to scale the binary T
    double scaleRoundErrors;        //TODO - IMPORTANT Value should depend on k
    boolean scaleRoundErrorsMul = false;

    public void setPhysicalStore()
    {
        doublePrimitiveDenseStoreFactory = PrimitiveDenseStore.FACTORY;
        //Actual allocation of matrix of size m x n, m: number of terms, n: number of documents
        this.m = m;
        this.n = n;
    }

    public int setm(long m)
    {
        this.m = m;
        return 0;
    }

    public int setn(long n)
    {
        this.n = n;
        return 0;
    }

    public int setk(long k)
    {
        this.k = k;
        return 0;
    }

    public long getm()
    {
        return m;
    }

    public long getn()
    {
        return n;
    }

    public long getk()
    {
        return k;
    }

    public double getScaleRoundErrors()
    {
        return scaleRoundErrors;
    }

    public int setScaleRoundErrors(double scaleRoundErrors)
    {
        this.scaleRoundErrors = scaleRoundErrors;
        return 0;
    }

    public PrimitiveDenseStore getZeroedOutPrimDensStore(long m, long n)
    {
        PrimitiveDenseStore primDensDouble = null;
        if ( doublePrimitiveDenseStoreFactory != null )
        {
            primDensDouble = doublePrimitiveDenseStoreFactory.makeZero(m, n);
        }
        else
        {
            System.err.println("Cannot allocate from physical store factory");
        }
        return primDensDouble;
    }

    public PrimitiveDenseStore getFilledOutPrimDensStore( MatrixStore<Double> a )
    {
        long rowCnt, colCnt, i, j;
        if ( a == null )
        {
            System.err.println("ERROR!!! Invalid parameter a:null!");
            return null;
        }

        rowCnt = a.countRows();
        colCnt = a.countColumns();

        PrimitiveDenseStore primitiveDenseStore = getZeroedOutPrimDensStore(rowCnt, colCnt);
        for ( i=0; i<rowCnt; i++ )
        {
            for ( j=0; j<colCnt; j++ )
            {
                primitiveDenseStore.set(i, j, a.get(i, j));
            }
        }
        return primitiveDenseStore;
    }
    public int addScalar( PrimitiveDenseStore a, double scalar )
    {
        long rowCnt, colCnt, i, j;
        if ( a == null )
        {
            System.err.println("ERROR!!! Invalid parameter a:null!");
            return -1;
        }

        rowCnt = a.countRows();
        colCnt = a.countColumns();

        for ( i=0; i<rowCnt; i++ )
        {
            for ( j=0; j<colCnt; j++ )
            {
                a.add(i, j, scalar);
            }
        }
        return 0;
    }

    //TODO create an interface for this class to incorporate binary matrix creation functionality
    public LSI_OjAlgo(int m, int n, boolean forBinaryVector)
    {
        //This constructor would only be called during server context
        this.clientContext = false;
        //Create a Physical store factory. Using this factory we will allocate space to store the original
        //high-dimensional term-document matrix C
        setPhysicalStore();
        setm(m);
        setn(n);
        setk(-1);   //Would be set and used later in this.computeKRankApproximation()
        //Whether the C matrix is going to be used for TFIDF matrix or a binary(term-document incidence) matrix
        this.forBinaryVector = forBinaryVector;
        //System.out.println("Binary LSI:"+forBinaryVector);
        //if (forBinaryVector)
        //{
        setScaleRoundErrors(1000);
        //}
        C = getZeroedOutPrimDensStore(m, n);

        //TODO: Release unused rows from C after k-estimation
    }

    public LSI_OjAlgo(long m, long n, long k, boolean forBinaryVector)
    {
        this.clientContext = true;
        setPhysicalStore();
        setm(m);
        setn(n);
        setk(k);
        U_k = null;
        //Whether the C matrix is going to be used for TFIDF matrix or a binary(term-document incidence) matrix
        this.forBinaryVector = forBinaryVector;
        //System.out.println("Binary LSI:"+forBinaryVector);
        //if (forBinaryVector)
        //{
        setScaleRoundErrors(1000);
        //}
        C = getZeroedOutPrimDensStore(m, 1);//C will store query vector as a column vector[m x 1]
    }

    public int buildU_kForClient(LinkedList<LinkedList<Double>> U_kAdjList)
    {
        int ret = 0;
        long rowCnt = 0, colCnt = 0;

        if (U_k == null)
        {
            U_k = getZeroedOutPrimDensStore(getm(), getk());
        }

        if ( (getm() != U_kAdjList.size()) || ( getk() != U_kAdjList.get(0).size() ) )
        {
            System.err.println("ERROR! Dimensions of U_k adj. list does not match with that of Primitive dense store " +
                    "matrix! ["+U_kAdjList.size()+", "+U_kAdjList.get(0).size()+"] != ["+getm()+", "+getk()+"]");
            return -1;
        }

        Iterator<LinkedList<Double>> rowIt= U_kAdjList.iterator();
        while ( rowIt.hasNext() )
        {
            LinkedList<Double> rowList = rowIt.next();
            Iterator<Double> doubleIt = rowList.iterator();
            colCnt = 0;
            while ( doubleIt.hasNext() )
            {
                U_k.set(rowCnt, colCnt, doubleIt.next());
                colCnt++;
            }
            rowCnt++;
        }
        //System.out.println("Populated prim. den. store U_k for lsi: "+U_k);
        SVDComputed = true;
        kRankApprox = true;

        return ret;
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
                if ( forBinaryVector == true )
                {
                    if ( termWeight == 0 )
                    {
                        //put a 0 against it
                        C.set(termNo, docNo, 0.0);
                    }
                    else
                    {
                        //put 1 instead of termWeight, indicating incidence between term and document
                        C.set(termNo, docNo, 1);
                    }
                }
                else
                {
                    C.set(termNo, docNo, termWeight);
                }
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
            return -1;
        }
        //else
        //{
            //System.out.println("C["+C.countRows()+" x "+C.countColumns()+"] matrix populated!!!");
        //}
        CFilled = true;
        return 0;
    }
    //Gets called only during client context
    int getReducedDimQuery()
    {
        int err = 0;
        //m dimensional query is in C
        if ( CFilled == false )
        {
            System.err.println("ERROR!!! Please fill the term-document matrix C");
            return -1;
        }
        if ( ! iskRankApprox() )
        {
            System.err.println("ERROR!!! U_k has not been filled-in. U_k originates from server! U_k:"+U_k);
            return -2;
        }
        if ( C.countColumns() != 1 )
        {
            System.err.println("ERROR!!! As of now only one query document is supported!");
            return -3;
        }
        if ( (C.countRows() != U_k.countRows()) )
        {
            System.err.println("ERROR! Dimensions do not match for matrix multiplication " +
                    returnDimString(C.countRows(), C.countColumns())+"^T X " +
                    returnDimString(U_k.countRows(), U_k.countColumns()));
            return -4;
        }
        MatrixStore<Double> c_trans = C.transpose();//MatrixStore<Double>
        T = c_trans.multiply(U_k);//dimensions should be 1 x k
        this.scaledupTermVectComputed = true;
        System.out.println("T:Query scaled computed forBinaryVector:"+forBinaryVector+":- Dimensions - Exp.:"+returnDimString(1, k)+" present:"+returnDimString(T));

        adjustTForCryptoOperations();

        //System.out.println("T matrix - min:"+getMin()+" max:"+getMax());
        //System.out.println("Before mul by roundoff constant T:"+T);
        /*
        if ( this.forBinaryVector == true )
        {
            //TODO - Check if the scaling up to binary works
            T = T.multiply(this.getScaleRoundErrors());
            System.out.println("Scaling up T by a factor of:"+ this.getScaleRoundErrors()+" for binary vector to avoid round off errors!");
        }
        System.out.println("After mul by roundoff constant T:"+T);*/


        return err;
    }

    public boolean isClientContext()
    {
        return clientContext;
    }

    public boolean isForBinaryVector()
    {
        return forBinaryVector;
    }

    public boolean isCFilled()
    {
        return CFilled;
    }

    public boolean isSVDComputed()
    {
        return SVDComputed;
    }

    public boolean iskRankApprox()
    {
        return kRankApprox;
    }

    public boolean isScaledupTermVectComputed()
    {
        return scaledupTermVectComputed;
    }

    public String returnDimString(long row, long col )
    {
        return "["+row+" x "+col+"]";
    }

    public String returnDimString( MatrixStore<Double> mat )
    {
        return returnDimString(mat.countRows(), mat.countColumns());
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
        //System.out.println("\n\niscomputed = "+svd.isComputed());
        svd.decompose(C);
        //System.out.println("\n\niscomputed = "+svd.isComputed());
        U = svd.getQ1();
        V = svd.getQ2();
        Sigma = svd.getD();

        //System.out.println("\n\nU = "+U);
        //System.out.println("\n\nSigma = "+Sigma);
        //System.out.println("\n\nV = "+V);


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
        this.setk(k);
        if ( this.getk() > Sigma.countRows() )
        {
            System.err.println("WARNING! current rank:"+Sigma.countRows()+" requested reduced rank k:"+k);
            return 0;
        }
        PhysicalStore SigmaApprox = Sigma.copy();

        //System.out.println("SigmaApprox:"+SigmaApprox);
        Sigma_k = truncateSigmaMemFriendly(k);
        U_k = truncateRightmostColumns((U.countColumns()-k), U);
        V_k = truncateRightmostColumns((V.countColumns()-k), V);
        kRankApprox = true;

        //Destroy unnecessary objects
        Sigma = null;
        U = null;
        V = null;

        String whetherBinary = "\t Is binary type?:"+forBinaryVector;
        System.out.println("k = "+k);
        System.out.println("Sigma_k dimensions:"+Sigma_k.countRows()+" x "+Sigma_k.countColumns()+whetherBinary);
        System.out.println("U_k dimensions:"+U_k.countRows()+" x "+U_k.countColumns()+whetherBinary);
        System.out.println("V_k dimensions:"+V_k.countRows()+" x "+V_k.countColumns()+whetherBinary);
        //After truncation, sizes of the matrices are:
        //Sigma_k: k x k
        //U_k: m x k
        //V_k: n x k
        //System.out.println("U_k:"+whetherBinary+U_k);
        //System.out.println("V_k:"+whetherBinary+V_k);

        //Computing the scaled-up document matrix (V_k x Sigma_k)
        T = V_k.multiply(Sigma_k);
        //MatrixStore T = V_k;
        this.scaledupTermVectComputed = true;
        System.out.println("T dimensions:"+T.countRows()+" x "+T.countColumns()+whetherBinary);

        /*DBG
        System.out.println(">>>>>>>>>>>>BEFORE");
        computePairwiseSimilarity(false);*/

        adjustTForCryptoOperations();

        //DBG
        /*System.out.println("<<<<<<<<<<<<AFTER");
        computePairwiseSimilarity(true);*/



        return 0;
    }

    /*This is a test function to see similarity of documents in collection in reduced dimensions*/
    public int computePairwiseSimilarity(boolean removeNegativeCorrection)
    {
        double correctionFactor=0;
        if ( (negativeCorrectionAdded == false) && (removeNegativeCorrection == true))
        {
            System.err.println("ERROR!!! Cannot remove negative correction as it has not been added in first place!");
            return -1;
        }
        long n = T.countRows(); //Store number of documents
        //Computing similarity
        System.out.println("============TESTING SIMILARITY SCORES Binary:"+isForBinaryVector()+"==============START");
        for ( int queryDocNum = 0; queryDocNum < n; queryDocNum++ )
        {
            //queryDocNum = 0 - first one
            Access1D<Double> query = T.sliceRow(queryDocNum, 0);
            System.out.println("\n\nqueryDocNum:" + queryDocNum + "\t\tquery:" + query);

            for (int i = 0; i < n; i++)
            {
                Access1D<Double> candidateDoc = T.sliceRow(i, 0);
                double score = candidateDoc.dot(query);
                if ( removeNegativeCorrection )
                {
                    Double sumOfRow = getRowSum(T, i);
                    if ( sumOfRow == null )
                    {
                        System.err.println("ERROR!!! Executing getRowSum()!");
                        return -2;
                    }
                    double negativeCorrectionValue = negativeCorrection;
                    if ( scaleRoundErrorsMul == true )
                    {
                        negativeCorrectionValue = negativeCorrectionValue * this.getScaleRoundErrors();
                    }
                    correctionFactor = (sumOfRow.doubleValue())*negativeCorrectionValue;
                }
                System.out.println("DocNum:" + i + "\tcandidate:" + candidateDoc +
                        "\nbeforeCorrectionScore:" + score+"\tafterCorrectionScore:"+(score-correctionFactor)+
                        "\tcorrectionFactor:"+correctionFactor+"\n");
            }

        }
        System.out.println("============TESTING SIMILARITY SCORES Binary:"+isForBinaryVector()+"================END\n\n\n\n\n");
        return 0;
    }

    public Double getRowSum(MatrixStore a, long rowIndex)
    {
        long colCount;
        double sum;
        if ( rowIndex >= a.countRows() )
        {
            System.err.println("ERROR!!! Index "+rowIndex+" out of bounds "+a.countRows());
            return null;
        }
        colCount = a.countColumns();
        sum = 0;
        for (long i = 0; i < colCount; i++)
        {
            sum = sum + a.get(rowIndex, i).doubleValue();
        }
        return new Double(sum);
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
        //System.out.println("Sigma--------->\n"+Sigma);
        //System.out.println("Sigma: limitOfColumn:"+Sigma.limitOfColumn(1000)+" limitOfRow:"+Sigma.limitOfRow(2000));

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


    public LinkedList getMatrixRow(String matrixType, long rowNo)
    {
        LinkedList rowInList;
        long noCol, i;
        switch (matrixType)
        {
            case "TruncatedU_k":

                if ( this.kRankApprox == false )
                {
                    System.err.println("ERROR!!! U_k not generated yet! Matrix approx. not performed yet!");
                    return null;
                }
                rowInList = new LinkedList<Double>();
                noCol = this.U_k.countColumns();
                if ( (rowNo<0) || (rowNo>= U_k.countRows()) )
                {
                    System.err.println("U_k[row:0, ...,"+(U_k.countRows()-1)+"] requested row:"+rowNo);
                    return null;
                }
                for ( i=0; i<noCol; i++ )
                {
                    rowInList.add(U_k.get(rowNo, i));
                }
                break;

            default:
                System.err.println("ERROR!!! Unrecognized matrix type");
                return null;
        }
        return rowInList;
    }
    /*For the given file name index, write the entire k dimensional row vectors to files*/
    public int writeRedDimVectTToFile(long rowIndex, long k, String vectFileName)
    {
        int err = 0;
        if ( ! isScaledupTermVectComputed() )
        {
            System.err.println("ERROR!!! Reduced dimension document matrix T not computed! No vectors present to write to file!");
            return -1;
        }
        if (( rowIndex >=  T.countRows()) || (rowIndex < 0) )
        {
            System.err.println("ERROR!!! Given rowIndex:"+rowIndex+" is out of bounds! Actual rows:"+T.countRows());
            return -2;
        }
        if ( (isClientContext()) && (rowIndex>=1) )
        {
            System.err.println("ERROR!!! Given rowIndex:"+rowIndex+" is out of bounds as only one query document is " +
                    "supported! Actual rows:"+T.countRows());
            return -3;
        }
        if ( k != T.limitOfRow((int)rowIndex) )
        {
            System.err.println("ERROR!!! Row(doc.) index:"+rowIndex+" has just "+T.limitOfRow((int)rowIndex)+" columns instead of "+k+" required!");
            return -4;
        }
        //File open
        File fp = new File(vectFileName);
        FileWriter fileWriter = null;
        try
        {
            fileWriter = new FileWriter(fp);

            //while(termStrIt.hasNext())
            for ( long i = 0; i < k; i++ )
            {
                //String term = termStrIt.next();
                String ifNewLine = "";
                //double weight = docWeightInfo.get(term);
                double weight = T.get(rowIndex, i);
                if ( (k != 1) && (i < k - 1) )
                {
                    ifNewLine = "\n";
                }
                fileWriter.write(((int) weight) + ifNewLine);
                //System.out.println("Added in " + TFIDFVectFileName + "::" + (count+1) + " " + term + ": " + weight);
                //System.out.println("Added in " + BinVectFileName + "::" + (count + 1) + " " + term + ": " + bin);
            }

            //Flush and close the file
            fileWriter.flush();
            fileWriter.close();

            //TODO: Remove debug statements below
            //printDocTermVectors(TFIDFVectFileName);
            //printDocTermVectors(BinVectFileName);
            //System.out.println("\n\n");
        } catch (IOException e)
        {
            e.printStackTrace();
        }

        return err;
    }

    public PrimitiveDenseStore normalizeMatrix(MatrixStore<Double> a)
    {
        long i, j, rowCnt, colCnt;
        double magnitude = 0, temp;

        if ( a == null )
        {
            System.err.println("ERROR!!! Invalid input parameter a == null!");
            return null;
        }

        rowCnt = a.countRows();
        colCnt = a.countColumns();

        //Create a new matrix
        PrimitiveDenseStore normalized = getZeroedOutPrimDensStore(rowCnt, colCnt);

        for (i=0; i<rowCnt; i++)
        {
            //Each row is a vector and we have to normalize each vector such that sum of square of each element after
            // normalization is 1.
            //for
            magnitude = 0;
            for ( j=0; j<colCnt; j++ )
            {
                temp = a.get(i, j);
                magnitude = magnitude + (temp*temp);
            }
            for ( j=0; j<colCnt; j++ )
            {
                temp = a.get(i, j);
                temp = temp / magnitude;
                normalized.set(i, j, temp);
            }
        }

        return normalized;
    }

    public long getMinMax(MatrixStore<Double> a)
    {
        long err = 0, i, j, rowCnt, colCnt;
        double temp;

        if ( a == null )
        {
            System.err.println("ERROR!!! Invalid input parameter a == null!");
            return -1;
        }

        rowCnt = a.countRows();
        colCnt = a.countColumns();

        min = a.get(0, 0);
        max = min;
        for (i=0; i<rowCnt; i++)
        {
            for ( j=0; j<colCnt; j++ )
            {
                temp = a.get(i, j);
                if ( temp < min )
                {
                    min = temp;
                }
                if ( temp > max )
                {
                    max = temp;
                }
            }
        }

        return err;
    }
    public double getMax()
    {
        return max;
    }
    public double getMin()
    {
        return min;
    }

    public double getNegativeCorrection()
    {
        return negativeCorrection;
    }

    /*This effective correction factor has to be subtracted from dot products in the C code*/
    public double getEffectiveCorrection()
    {
        double negCorrection = 0;
        if ( negativeCorrectionAdded == true )
        {
            if ( scaleRoundErrorsMul == true )
            {
                negCorrection =  (getNegativeCorrection()*getScaleRoundErrors());
            }
            else
            {
                negCorrection = (getNegativeCorrection());
            }
        }
        return negCorrection;
    }

    /* Output is a double rounded to its floor value. Basically if you have two docs
    d1: a,b,c
    d2: d,e,f
    negativeCorrection is n
    and roundCorrection is s
    and k = 3 for three dimensions.
    on neg cor you get
    d1: a+x,b+x,c+x
    d2: d+x,e+x,f+x
    on round correction,
    d1: s(a+x),s(b+x),s(c+x)
    d2: s(d+x),s(e+x),s(f+x)
    on dot product you get
    s^2(ad+be+cf)+s^2(a+b+c)n+s^2(3n^2)+s^2(d+e+f)n
    if d1 is a constant query then s^2(d+e+f)n is the term that will skew the original dot product ad+be+cf so we need
    to subtract it, rest of the terms act as constant as query is constant.

    Now input in T is
    d1: s(a+x),s(b+x),s(c+x)
    d2: s(d+x),s(e+x),s(f+x)
    below function computes the magnitude of the term to be subtracted i.e. s^2(d+e+f)n. This would be then subtracted
    using the C code once encrypted, intermediate dot products have been computed for tfidf and bin respectively.
    Although the return type is double, it has been rounded to its ceiling value.
    * */
    public double getRowEffectiveCorrection(long docIndex)
    {
        double rowSum = 0, correction = 0;
        if ((docIndex >= T.countRows()) || (docIndex<0))
        {
            System.err.println("ERROR!!! Document index out of bounds!");
            return -1;
        }

        rowSum = getRowSum(T, docIndex);//of form: s(sumOrigRowsB4AnyCorrectn)+s*k*n

        correction = 0;
        if ( negativeCorrectionAdded == true )
        {
            correction = getNegativeCorrection();//Mul by n
        }

        if ( scaleRoundErrorsMul == true )
        {
            correction = correction * getScaleRoundErrors();//Mul by s
        }
        correction = correction * T.countColumns();//Mul by k

        if ( rowSum < correction )
        {
            //System.err.println("WARNING!!! In correction computation, docIndex:"+docIndex);
        }
        rowSum = rowSum - correction;//of form:s(sumOrigRowsB4AnyCorrectn)

        if ( negativeCorrectionAdded == true )
        {
            correction = rowSum * getNegativeCorrection();//s*(sumofRowB4Ctn)*n
            if ( scaleRoundErrorsMul == true )
            {
                correction = correction * getScaleRoundErrors();//s*(sumOfRowB4Ctn)*n*s
            }
        }
        else
        {
            correction = 0;
        }
        return (Math.floor(correction));

    }

    public int adjustTForCryptoOperations()
    {
        double min, max, absMin, absMax;


        /*
        //Normalize
        PrimitiveDenseStore temp;
        System.out.println("T before normalization "+T);
        temp = normalizeMatrix(T);
        if ( temp == null )
        {
            System.err.println("ERROR!!! In normalizing T matrix");
            return -1;
        }
        System.out.println("T after normalization "+temp);
        T = temp.get();*/

        //Remove negative elements
        getMinMax(T);
        min = getMin();
        max = getMax();

        if ( min < 0 )
        {
            absMin = -1 * min;
        }
        else
        {
            absMin = min;
        }

        if ( max < 0 )
        {
            absMax = -1 * max;
        }
        else
        {
            absMax = max;
        }
        this.negativeCorrection = absMin;
        //System.out.println("T before adjusting for negative correction "+T);
        PrimitiveDenseStore primitiveDenseStore = getFilledOutPrimDensStore(T);
        if ( min < 0 )
        {
            addScalar(primitiveDenseStore, negativeCorrection);
            negativeCorrectionAdded = true;
        }
        T = primitiveDenseStore.get();
        //System.out.println("T after adjusting for negative correction "+T);

        //For binary vectors where elements can have values between 0 to 1, we scale them up. TFIDF does not require this - Modified it does require here -
        // as their values are already huge due to scaling done while generating the vectors from index see getDocTFIDFVectors()
        // for normalizeAndScaleUpVector() function.
        //System.out.println("Before mul by roundoff constant T:"+this.forBinaryVector+" "+T);
        //if ( this.forBinaryVector == true )
        //{
        T = T.multiply(this.getScaleRoundErrors());
        scaleRoundErrorsMul = true;
        //System.out.println("Scaling up T by a factor of:"+ this.getScaleRoundErrors()+" for binary vector to avoid round off errors!");
        //}

        //System.out.println("T:"+this.forBinaryVector+" "+T);
        return 0;
    }
}
