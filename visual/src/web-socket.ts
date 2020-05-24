import { WebSocketClient, GraphQLSubscription } from './client.js';

async function ws() {
    let client = await WebSocketClient('ws://localhost:8081');
    console.log('UI');
    // let client2 = await WebSocketClient('ws://localhost:8081');
    // let i = 0;
    // while(++i) {
    //     await sleep(500);
    //     await client.put(i);
    //     // Nice, now I have seletable web socket connections
    //     // Now just need to implement a shuffle algorithm for selection fairness
    //     let x = await client.pop();
    //     console.log('pop', x);
    // }

    // let subscription = await GraphQLSubscription(`subscription {hello}`, client);
    // while (i) {
    //     console.log(1, await subscription.pop());
    //     console.log(2, await subscription2.pop());
    //     let subscription2 = await GraphQLSubscription(`mutation { hello(text: "${i}") }`, client2);
    // }
}