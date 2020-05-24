// @ts-nocheck
import { chan, select } from 'https://creatcodebuild.github.io/csp/dist/csp.js';
function SortVisualizationComponent(id, arrays) {
    let ele = document.getElementById(id);
    let stop = chan();
    let resume = chan();
    // Animation SVG
    CreateArrayAnimationSVGComponent(ele, id + 'animation', 0, 0)(arrays, stop, resume);
    // Stop/Resume Button
    let button = ele.getElementsByTagName('button')[0];
    let stopped = false;
    button.addEventListener('click', async () => {
        // if(!clicked) {
        console.log('clicked', stopped, '->', !stopped);
        stopped = !stopped;
        if (stopped) {
            button.textContent = 'resume';
            await stop.put(null);
        }
        else {
            button.textContent = 'stop';
            await resume.put(null);
        }
    });
}
function CreateArrayAnimationSVGComponent(parent, id, x, y) {
    let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.id = id;
    let div = document.createElement('div');
    div.appendChild(svg);
    parent.insertBefore(div, parent.firstChild);
    return async (arrays, stop, resume) => {
        let waitToResume = await needToStop(stop, resume);
        for await (let array of arrays) {
            await waitToResume.pop();
            while (svg.lastChild) {
                svg.removeChild(svg.lastChild);
            }
            for (let [i, number] of Object.entries(array)) {
                let r = rect(x + Number(i) * 4, y, 3, number);
                svg.appendChild(r);
            }
            await sleep(300);
        }
    };
    function rect(x, y, width, height) {
        // https://developer.mozilla.org/en-US/docs/Web/API/Document/createElementNS
        // https://stackoverflow.com/questions/12786797/draw-rectangles-dynamically-in-svg
        let rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('width', width);
        // @ts-ignore
        rect.setAttribute('height', height);
        // @ts-ignore
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        // rect.classList.add(className);
        return rect;
    }
}
function sleep(time) {
    return new Promise((resolve) => {
        setTimeout(resolve, time);
    });
}
async function InsertionSort(array, reactor) {
    function insert(array, number) {
        // [1, 2, 4, 5], 3
        // in-place
        // immutable 不可变
        if (array.length === 0) {
            return [number];
        }
        let sorted = [];
        let inserted = false;
        for (let i = 0; i < array.length; i++) { // n
            if (!inserted) {
                if (number < array[i]) {
                    inserted = true;
                    sorted.push(number);
                }
            }
            sorted.push(array[i]);
        }
        if (!inserted) {
            sorted.push(number);
        }
        return sorted;
    }
    let sortedArray = [];
    for (let i = 0; i < array.length; i++) { // n
        sortedArray = insert(sortedArray, array[i]);
        await reactor.put(sortedArray.concat(array.slice(i + 1)));
    }
    return sortedArray;
}
async function MergeSort(array, reactor) {
    async function merge(l, r, startIndex) {
        if (l.length === 0) {
            return r;
        }
        if (r.length === 0) {
            return l;
        }
        let shifted = await (async () => {
            if (l[0] < r[0]) {
                return l.slice(0, 1).concat(await merge(l.slice(1), r, startIndex + 1));
            }
            else {
                return r.slice(0, 1).concat(await merge(l, r.slice(1), startIndex + 1));
            }
        })();
        // console.log(shifted, startIndex)
        await reactor.put([shifted, startIndex]);
        return shifted;
    }
    async function sort(array, startIndex) {
        if (array.length <= 1) {
            return array;
        }
        let m = Math.floor(array.length / 2);
        let l = array.slice(0, m);
        let r = array.slice(m);
        let sortedL = await sort(l, startIndex);
        let sortedR = await sort(r, startIndex + m);
        await reactor.put([sortedL.concat(sortedR), startIndex]);
        // need global index here to correctly animate
        let merged = await merge(sortedL, sortedR, startIndex);
        await reactor.put([merged, startIndex]);
        return merged;
    }
    await reactor.put([array, 0]);
    return await sort(array, 0);
}
function controlButton(stop, resume) {
    let button = document.getElementById('controlButton');
    let stopped = false;
    button.onclick = async () => {
        // if(!clicked) {
        console.log('clicked', stopped, '->', !stopped);
        stopped = !stopped;
        if (stopped) {
            await stop.put(null);
        }
        else {
            console.log('resume');
            await resume.put(null);
        }
    };
}
async function main() {
    // init an array
    let array = [];
    for (let i = 0; i < 50; i++) {
        array.push(Math.floor(Math.random() * 50));
    }
    // event queue
    let insertQueue = chan();
    let mergeQueue = chan();
    let stop = chan();
    let resume = chan();
    // controlButton(stop, resume);
    console.log('begin sort', array);
    let s1 = InsertionSort(array, insertQueue);
    let s2 = MergeSort(array, mergeQueue);
    console.log('after sort');
    let mergeQueue2 = (() => {
        let c = chan();
        (async () => {
            let numebrsToRender = [].concat(array);
            await c.put(numebrsToRender);
            while (1) {
                let [numbers, startIndex] = await mergeQueue.pop();
                // console.log(numbers);
                for (let i = 0; i < numbers.length; i++) {
                    numebrsToRender[i + startIndex] = numbers[i];
                }
                await c.put(numebrsToRender);
            }
        })();
        return c;
    })();
    console.log(mergeQueue2);
    // SortVisualizationComponent('insertion-sort', insertQueue);
    // SortVisualizationComponent('merge-sort', mergeQueue2);
    UI(stop, resume, array);
}
main();
async function needToStop(stop, resume) {
    let stopResume = chan();
    let stopped = false;
    (async () => {
        while (1) {
            await select([
                [resume, async () => {
                        stopped = false;
                        await stopResume.put();
                    }],
                [stop, async () => {
                        stopped = true;
                    }]
            ], async () => {
                if (stopped) {
                    await resume.pop();
                    stopped = false;
                }
                else {
                    await stopResume.put();
                }
            });
        }
    })();
    return stopResume;
}
function UI(stop, resume, array) {
    Vue.component('sort-visualization', {
        props: ['name'],
        data() {
            return {
                stopped: false,
                state: 'stop',
                array: []
            };
        },
        methods: {
            async click(event) {
                this.stopped = !this.stopped;
                if (this.stopped) {
                    this.state = 'resume';
                    await stop.put(null);
                }
                else {
                    console.log(this.array);
                    this.state = 'stop';
                    await resume.put(null);
                }
            }
        },
        template: `
        <div>
        <div> {{name}} </div>
        <button v-on:click="click"> {{ state }} </button>
        <svg>
            <rect v-for="i in array" x="120" width="5" height="10" rx="15" />
        </svg>
        </div>
        `
    });
    console.log('UI');
    console.log(array);
    let iSort = new Vue({
        el: '#insertion-sort'
    });
    iSort.array = array;
    new Vue({ el: '#merge-sort' });
}
