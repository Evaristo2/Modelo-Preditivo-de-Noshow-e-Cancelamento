# Modelo-Preditivo-de-Noshow-e-Cancelamento
Nas empresas de aluguel de carro, o termo noshow significa quando o cliente faz a reserva do carro mas quando chega o dia de retirar da loja, ele não aparece. 
Isso é Noshow.
O meu modelo faz uma análise dos clientes e verifica qual tem a probabilidade de noshow ou cancelar a reserva sem aparecer na loja.
O algoritmo escolhido foi o RandomForest. Como feature, foi utilizado o score do cliente, a modalidade da reserva (se é diária ou mensal) e o canal em que o cliente fez a reserva.

com isso, chegamos no resultado seguinte de matriz confusão:
![image](https://github.com/user-attachments/assets/b97fc0ee-a8c3-4746-846f-8ec54df5dd7d)

O modelo está em produção na minha empresa e atua diretamente em um robo, feito no BigQuery com Procedures, que faz a disponibilidade da frota.
