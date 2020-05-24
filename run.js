class Channel {
    constructor() {
        this.popActions = [];
        this.putActions = [];
    }

    put(ele) {
        // if no pop action awaiting
        if (this.popActions.length === 0) {
            return new Promise((resolve) => {
                console.log('put promise');
                if (this.putActions.length >= 1) {
                    throw new Error('all promise asleep');
                }
                this.putActions.push([resolve, ele]);
            })
        } else {
            let onPop = this.popActions.shift();
            onPop(ele);
            return new Promise((resolve) => {
                resolve()
            });
        }
    }

    pop(ele) {
        if (this.putActions.length === 0) {
            return new Promise((resolve) => {
                console.log('pop promise');
                if (this.popActions.length >= 1) {
                    throw new Error('pop all promise asleep');
                }
                this.popActions.push(resolve);
            })
        } else {
            let [onPut, ele] = this.putActions.shift();
            onPut();
            return new Promise((resolve) => {
                resolve(ele)
            });
            // return ele;
        }
    }
}

let c = new Channel();
async function main() {
    let thread = async () => {
        while (1) {
            await c.put(1);
            await c.put(2);
        }
    }
    thread();
    while (1) {
        let x = await c.pop();
        console.log('B', x);
    }
    console.log('end');
}
main();


let result = await select(
    c, async (ele) => {
        return ele;
    },
    timeout(100), async () => {
        return 'timed out'
    }
)

