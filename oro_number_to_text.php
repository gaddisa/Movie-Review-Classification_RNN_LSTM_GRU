<?php

function convert_number($number) 
{
    $Number_Decimal=explode('.',$number);
    $number=str_replace(',','',$Number_Decimal[0]); 
    $rt_1=count($Number_Decimal);
    if (($number < 0) || ($number > 9999999999)) {
        throw new Exception("Number is out of range");
    }

    $Bn = floor($number / 1000000000);
    /* Billions */
    $number -= $Bn * 1000000000;
    $Gn = floor($number / 1000000);
    /* Millions */
    $number -= $Gn * 1000000;
    $kn = floor($number / 1000);
    /* Thousands */
    $number -= $kn * 1000;
    $Hn = floor($number / 100);
    /* Hundreds */
    $number -= $Hn * 100;
    $Dn = floor($number / 10);
    /* Tens */
    $n = $number % 10;
    /* Ones */

    $res = "";

    if ($Bn) {
        $res .= " "."Biliyoona ".convert_number($Bn);
    }
 
    if ($Gn) {
        $res .=" "."Miliyoona ".convert_number($Gn);
    }

    if ($kn) {
        $res .=" "."Kuma"." ".convert_number($kn).(empty($res) ? "" : " ");
    }

    if ($Hn) {
        $res .= " "."Dhibba".(empty($res) ? "" : " ")." " .convert_number($Hn);
    }

    $ones = array("", "Tokko", "Lama", "Sadii", "Afur", "Shan", "Jaha", "Torba", "Saddeet", "Sagal", "Kudhan", "Kudha-tokko", "Kudha-lama", "Kudha-sadii", "Kudha-afur", "Kudha-shan", "Kudha-jaha", "Kudha-torba", "Kudha-saddet", "Kudha-sagal");
    $ones_ext = array("", "tokko", "lama", "sadii", "afur", "shan", "jaha", "torba", "saddeet", "sagal", "Kudhan", "Kudha-tokko", "Kudha-lama", "Kudha-sadii", "Kudha-afur", "Kudha-shan", "Kudha-jaha", "Kudha-torba", "Kudha-saddet", "Kudha-sagal");

    $tens = array("", "", "Digdama", "Soddoma", "Afurtama", "Shantama", "Jaatama", "Torbaatama", "Saddettama", "Sagaltama");

    if ($Dn || $n) {
        if (!empty($res)) {
            $xx=substr($res,-1);
            if($xx=='n'|| $xx=='r' || $xx=='t'){
                $xx.="ii";
                $res=substr_replace($res,$xx,-1);    
                $res .=" fi ";
            }
            else{
                $res .= $xx." fi ";
            }  
        }

        if ($Dn < 2) {
            $res .= $ones[$Dn * 10 + $n];
        } else {
            $res .= $tens[$Dn];

            if ($n) {  
                $tt=substr($res,-2);
                if($tt=="ma")
                    $res=substr_replace($res,"mii",-2);          
                $res .= "-" . $ones_ext[$n];
            }
        }
    }

    if (empty($res)) {
        $res = "Duwwaa";
    } 
    
    if($rt_1==2)
    {
        $number_dc=$Number_Decimal[1];
    }
    
     
    if($rt_1==2){
        $array  = array_map('intval', str_split($Number_Decimal[1]));
     
        $dec_read='';
        for($i=0; $i<count($array);$i++)
        {
            $dec_read.=' '.convert_number($array[$i]);
        } 
        return $res. " fii Saantima".$dec_read;
    }else{
        return $res;
    }
    
    $output = process_string($res);
    // Replace two spaces with one space
    $output = preg_replace('/\s{2,}/', ' ', $output);
	// Remove spaces before and after hyphens
	$res = preg_replace('/\s*-\s*/', '-', $output);

    return $res;
}
    

function process_string($input) {
    // Explode the input string into an array of words
    $words = explode(" ", $input);
    
    // Check if "Saantima" is present in the array
    $index = array_search("Saantima", $words);
    if ($index !== false && isset($words[$index + 1]) && isset($words[$index + 2])) {
        // Check if there are at least two more words after "Saantima"
        $next_word = $words[$index + 1];
        $next_next_word = $words[$index + 2];
        
        // Check if the next word is "Tokko"
        if ($next_word == "Tokko") {
            // Replace "Tokko" with "Kudha" and add a hyphen followed by the last word
            $words[$index + 1] = "Kudha-";
            $words[$index + 2] = $next_next_word;
        }
        // Check if the next word is "Lama"
        elseif ($next_word == "Lama") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Digdamii-";
            $words[$index + 2] = $next_next_word;
        }
        elseif ($next_word == "Sadii") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Soddomii-";
            $words[$index + 2] = $next_next_word;
        }
        elseif ($next_word == "Afur") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Afurtamii-";
            $words[$index + 2] = $next_next_word;
        }
         elseif ($next_word == "Shan") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Shantamii-";
            $words[$index + 2] = $next_next_word;
        }
        
         elseif ($next_word == "Jaha") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Jaatamii-";
            $words[$index + 2] = $next_next_word;
        }
         elseif ($next_word == "Torba") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Torbaatamii-";
            $words[$index + 2] = $next_next_word;
        }
         elseif ($next_word == "Saddet") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Saddeettamii-";
            $words[$index + 2] = $next_next_word;
        }
         elseif ($next_word == "Sagal") {
            // Replace "Lama" with "Digdama" and add a hyphen followed by the last word
            $words[$index + 1] = "Sagaltamii-";
            $words[$index + 2] = $next_next_word;
        }
    }
    
    elseif ($index !== false && isset($words[$index + 1])) {
         $next_word = $words[$index + 1];
        // Handle the case where there's only one word after "Saantima"
        // Replace with your logic here
        // For example:
        if ($next_word == "Tokko") {
            // Replace "Tokko" with "Kudhan"
            $words[$index + 1] = "Kudhan";
        } elseif ($next_word == "Lama") {
            // Replace "Lama" with "Digdama"
            $words[$index + 1] = "Digdama";
        } elseif ($next_word == "Sadii") {
            // Replace "Sadii" with "Soddoma"
            $words[$index + 1] = "Soddoma";
        } elseif ($next_word == "Afur") {
            // Replace "Afur" with "Afurtama"
            $words[$index + 1] = "Afurtama";
        } elseif ($next_word == "Shan") {
            // Replace "Shan" with "Shantama"
            $words[$index + 1] = "Shantama";
        } elseif ($next_word == "Jaha") {
            // Replace "Jaha" with "Jaatama"
            $words[$index + 1] = "Jaatama";
        } elseif ($next_word == "Torba") {
            // Replace "Torba" with "Torbaatama"
            $words[$index + 1] = "Torbaatama";
        } elseif ($next_word == "Saddet") {
            // Replace "Saddet" with "Saddeettama"
            $words[$index + 1] = "Saddeettama";
        } elseif ($next_word == "Sagal") {
            // Replace "Sagal" with "Sagaltama"
            $words[$index + 1] = "Sagaltama";
        }
    }
    
    // Implode the array of words back into a string
    $result = implode(" ", $words);
    return $result;
}


?>
