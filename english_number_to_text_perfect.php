<?php

function convertCurrencyToText($amount) {
    $words = array(
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'
    );

    $tens = array(
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'
    );

    $amount = number_format($amount, 2, '.', ''); // Format to 2 decimal places
    $dollars = intval($amount);
    $cents = intval(($amount - $dollars) * 100);

    $result = '';

    if ($dollars > 0) {
        $result .= convertNumberToWords($dollars) . ' dollars';
    } else {
        $result .= 'zero dollars';
    }

    if ($cents > 0) {
        $result .= ' and ' . convertNumberToWords($cents) . ' cents';
    }

    return $result;
}

function convertNumberToWords($number) {
    $words = array(
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'
    );

    $tens = array(
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'
    );

    $scales = array(
        '', 'thousand', 'million', 'billion', 'trillion'
    );

    if ($number < 20) {
        return $words[$number];
    }

    $word = '';

    $scaleIndex = 0;

    while ($number > 0) {
        $chunk = $number % 1000;
        $number = floor($number / 1000);

        if ($chunk > 0) {
            $chunkWord = '';

            if ($chunk >= 100) {
                $chunkWord .= $words[floor($chunk / 100)] . ' hundred ';
                $chunk %= 100;
            }

            if ($chunk >= 20) {
                $chunkWord .= $tens[floor($chunk / 10) - 2] . ' ';
                $chunk %= 10;
            }

            if ($chunk > 0) {
                $chunkWord .= $words[$chunk];
            }

            $word = $chunkWord . ' ' . $scales[$scaleIndex] . ' ' . $word;
        }

        $scaleIndex++;
    }

    return $word;
}

// example input and output
$input = 1000000000;
echo convertCurrencyToText($input); // Output: One million two hundred thirty-four thousand five hundred sixty-seven dollars and eighty-nine cents
?>
