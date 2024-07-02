from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response

from app.datatalk import DataTalk

# Create your views here.

pdfpath = r'C:\Users\Samruddhi More\Desktop\LLM BE Template\data\Voliro_Report-VOLAT-T-851-B-ANGLED.pdf'

class Chat(APIView):

    def post(self, request):
        chat = DataTalk(pdfpath)
        query = self.request.data.get('query')

        if query:


            response = chat.generate_response(query)

            return Response(response)




