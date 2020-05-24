function mergeSort(array) {
    if (array.length <= 1) {
        return array;
    }
    // left, right
    let [l, r] = split(array);
    let sortL = mergeSort(l);
    let sortR = mergeSort(r);
    return mergeRecursion(sortL, sortR);
}

// O(n^2)

// θ(n * log(n)) = merge cost * recursion count

function mergeLoop(l, r) {  // O() = 3 * (N + M) = O(N+M)
    let merged = [];
    let i = 0;  // l
    let j = 0;  // r
    while (merged.length < (l.length + r.length)) {
        // Left is done
        if (i === l.length) {
            merged.push(r[j])
            j++;
        }
        // Right is done
        else if (j === r.length) {
            merged.push(l[i])
            i++;
        }
        else if (l[i] < r[j]) {
            merged.push(l[i])
            i++;
        }
        else {
            merged.push(r[j])
            j++;
        }
    }
    return merged;
}

function mergeRecursion(l, r) {
    if (l.length === 0) {
        return r;
    }
    if (r.length === 0) {
        return l;
    }
    if (l[0] < r[0]) {
        let remaining = mergeRecursion(l.slice(1), r)
        return [l[0]].concat(remaining);
    } else {
        let remaining = mergeRecursion(l, r.slice(1))
        return [r[0]].concat(remaining);
    }
}

function split(array) {
    // [a, b, c]
    //  0, 1, 2
    // [a, b, c, d]
    //  0, 1, 2, 3
    let middle = Math.floor(array.length / 2);
    let l = array.slice(0, middle)
    let r = array.slice(middle)
    return [l, r]
}


console.log(mergeSort([1, 2, 3, 4, 5, 6]));
console.log(mergeSort([]));
// console.log(mergeSort([1]));
console.log(mergeSort([6, 5, 4, 3, 2, 1]));
console.log(mergeSort([2343, 4, 3, 4, 4, 23, 4, 345345]));
