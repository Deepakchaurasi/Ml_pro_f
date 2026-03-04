import queue as q
import time
import os
list1={
        'cake':100,
        'bread':50,
        'pastry':150,
        'cookie':20,
        'tee':30,
    }
class bakery:

    def __init__(self):
        self.__q_my=q.Queue()

    def addorder(self):
        ored_id=int(input("enter the custemer id\n"))
        order_name=input("enter the custmer name\n")
        order_item=[]
        l=input("enter the items to order (comma separated)\n").split(",")
        for temp in l:
             order_item.append(temp)
        num_item=len(order_item)
        order_time=time.ctime()
        order_bill=0

        for item in order_item:
            if item not in list1:
                print(f"{item} is not available in the bakery")
                return
            
            order_bill += list1[str(item)]
           

        order={
            "id":ored_id,
            "name":order_name,
            "items":order_item,
            "num_items":num_item,
            "time":order_time,
            "bill":order_bill
        }
        self.__q_my.put(order)
        print("****************order added successfully************************\n" )
        return
    def showorder(self):
        if self.__q_my.empty():
            print("************no any order************\n")
        else:
            print("\n******orders:************\n")
            for order in list(self.__q_my.queue):
                print(f"ID    |: {order['id']}, \n Name |: {order['name']}, \nItems |: {order['items']}, \nnum_items|: {order['num_items']},\norder_time|: {order['time']}\nBill  |: {order['bill']}")
        return
    def total_orders(self):
        print(f"\n******total orders: {self.__q_my.qsize()}\n")
        return 
    

    def removeorder(self):   
         if self.__q_my.empty():
            print("************no any order to remove************\n")
         else:
            order_id=int(input("enter the order id to remove\n"))
            found=False
            for order in list(self.__q_my.queue):
                if order['id']==order_id:
                    self.__q_my.get(order)
                    print("order removed successfully\n")
                    found=True
                    break
            if not found:
                print("order not found***********\n")


if __name__=="__main__":  
    customer=bakery()
    if os.name=="nt":
        os.system("cls")
    while True:
        print("1.add order\n2.show order\n3.total orders\n4.remove order\n")

        n=int(input("enter the number\n"))
        match(n):
            case 1:
                print("available items:\n")
                print("_____________________________")
                for key,val in list1.items():
                    print(f"|{key}:{val}|")
                customer.addorder()
                
            case 2:
                customer.showorder()
            case 3:
                customer.total_orders()
            case 4:
                customer.removeorder()
            case _:
                print("invalid input")
                exit()

