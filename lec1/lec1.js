function shuffle1(array) {
    for (let i = 0; i < array.length; i++) {
        let pick = Math.floor(Math.random() * (array.length)) // [0 ~ 1)
        swap(array, i, pick)
    }
    return array;
}
function shuffle2(array) { // O(n)
    for (let i = 0; i < array.length; i++) {
        let pick = Math.floor(Math.random() * (array.length - i)) // [0 ~ 1)
        swap(array, i, i + pick)
    }
    return array;
}
function shuffle3(remaining) {  // O(n^2)
    function x(shuffled, remaining) {
        if(remaining.length === 0) {
            return shuffled;
        }
        let pick = Math.floor(Math.random() * (remaining.length)) // [0 ~ 1)
        shuffled.push(remaining[pick])
        remaining.splice(pick, 1) // O(n) --> 0(1)
        return x(shuffled, remaining);
    }
    return x([], remaining)
}

function swap(array, i, j) {
    let temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

function test(f, M) {
    let results = [];
    for (let i = 0; i < M; i++) {
        let array = f([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        results.push(array)
    }

    let stats = [];
    for (let i = 0; i < 10; i++) {
        stats.push([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
    for (let result of results) {
        for (let j = 0; j < result.length; j++) {
            // j // 2
            // result[j] // 3
            // console.log(stats, j, stats[j]);
            if (stats[j][result[j]] === undefined) {
                stats[j][result[j]] = 0;
            } else {
                stats[j][result[j]] += 1;
            }
        }
    }
    return stats;
}

function d(stats, M) {
    M = M / 100
    let sum = 0;
    for (let row of stats) {
        for (let num of row) {
            sum += num / M
        }
    }
    let avg = sum / 100;

    let d_sum = 0;
    for (let row of stats) {
        for (let num of row) {
            d_sum += (num / M  - avg) * (num / M - avg)
        }
    }
    return d_sum / 100;
}

for(let i = 0; i < 5; i++) {
    let M = Math.pow(10, i);
    let t1 = test(shuffle1, M)
    let t2 = test(shuffle2, M)
    let t3 = test(shuffle3, M)
    console.log(d(t1, M), d(t2, M),  d(t3, M))
}





// console.log(shuffle3([1,2,3,4,5,6,7,8,9,10]));