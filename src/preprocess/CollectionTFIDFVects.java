package preprocess;

import preprocess.DocMagnitudeTreeMap;

import java.util.Set;
import java.util.TreeMap;

/**
 * Created by nuplavikar on 12/29/16.
 */
public class CollectionTFIDFVects extends TreeMap<String, DocMagnitudeTreeMap>
{
    //This function is important as files should always be in the same order especially for LSI as we use matrix
    //columns for representing documents. This function makes sure that they are always used in a fixed order, unless
    //there is any modification. Hence, whenever file names have to be accessed from a collection, they should be
    //done using this function to maintain consistency.
    Set<String> getSetOfFileNames()
    {
        return this.keySet();
    }

    DocMagnitudeTreeMap getDocVector(String fileName)
    {
        return this.get(fileName);
    }
}
