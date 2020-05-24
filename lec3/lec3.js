class LinkedList {
    constructor() {
        this.head = null;
        this._length = 0;
    }

    // 得到链表长度
    // O(1)
    length() {
        return this._length;
    }     
    
    // N --> O(n)
    // 在链表末尾追加元素
    append(data) {
        if(!this.head) {
            this.head = new Node(data);
        } else {
            this.head.append(data)
        }
        this._length += 1;
    }        
    // 在链表头部追加元素
    append_head(data) {
        let newHead = new Node(data)
        newHead.next = this.head;
        this.head = newHead;
        this._length += 1;
    }
    
    // 在链表 index-1 和 index 之间插入一个元素
    // insert(element, 0) 等于 append_head(element)
    // insert(element, length()) 等于 append(element)
    insert(element, index){

    }  
    
    // 得到 index 位的元素
    get(index) {
        if(index >= this.length()) {
            throw Error('not found');
        }
        return this.head.get(index);
    }      
}

class Node {
    constructor(data) {
        this.data = data
        this.next = null; // Node
    }

    append(data) {
        if(this.next) {
            return this.next.append(data)
        }
        this.next = new Node(data)
    }

    get(index) {
        console.log(this.data);
        if( index === 0) {
            return this.data;
        }
        return this.next.get(index-1);
    }
}


let list = new LinkedList();
list.append(1);
list.append(2);
list.append(3);
list.append_head(4);
list.append_head(5);
list.append_head(6);
// 6 5 4 1 2 3
list.get(5);
console.log('len', list.length());


// coupling（结对）    耦合：不该在一起的在一起，孽缘
// cohesion（和谐）    聚合：该在一起
// length 长度





