import os
import pandas as pd
from ..extractors.pdf_utils import PDF
from ..normalisation import normalize_string

def extract_ground_truth_docbank(pdf:PDF, label:str, tool:str) -> list[dict]:
    results = []
    
    for page_num, df in pdf.txt_pages.items():
        # Filter rows matching the target label and where token is not a layout marker
        filtered = df[(df['label'] == label) & (df['token'] != '##LTLine##')]

        if not filtered.empty:
            tokens = filtered['token'].tolist()
            combined_text = normalize_string(' '.join(map(str, tokens)))
            results.append({
                'tool' : tool,
                'pdf_name': pdf.pdf_name,
                'page': 0,
                'label': label,
                'data_gt': combined_text
            })
    return results

def extract_ground_truth_json(pdf:PDF, ex_label:str, tool:str) -> list[dict]: 
    results = []
    for key, value in pdf.txt_pages.items():
        if value == "" : continue
        # Treat complexe structures
        if ex_label == "author" and key == "author" : 
            for author in value :
                results.append({
                    'tool' : tool,
                    'pdf_name': pdf.pdf_name,
                    'page': 0,
                    'label': ex_label,
                    'data_gt': normalize_string( author['author_name'] )
                })
        
        elif ex_label == "affiliation" and key == "author" : # Affiliations are under the "authors" tag
            for author in value :
                for affiliation in author['affiliation'] : # Affilation is a list
                    results.append({
                        'tool' : tool,
                        'pdf_name': pdf.pdf_name,
                        'page': 0,
                        'label': ex_label,
                        'data_gt': normalize_string( affiliation )
                    })

        elif ex_label == "email" and key == "author" : # Emails are under the "authors" tag
            for author in value :
                emails = author.get('email', [])
                # if it's a lone string, wrap it in a list
                if isinstance(emails, str):
                    emails = [emails]
                for email in emails:
                    if not email:
                        continue
                    results.append({
                        'tool'    : tool,
                        'pdf_name': pdf.pdf_name,
                        'page'    : 0,
                        'label'   : ex_label,
                        'data_gt' : normalize_string( email )
                    })

        elif ex_label == "equation" and key == "equation":
            for eq in value :
                if eq : 
                    eq = eq.replace("_", " ").replace("^", " ")
                results.append({
                    'tool' : tool,
                    'pdf_name': pdf.pdf_name,
                    'page': 0,
                    'label': ex_label,
                    'data_gt': normalize_string( eq )
                })

        elif ex_label == "abstract" and key == "abstract":
            for abs in value :
                results.append({
                    'tool' : tool,
                    'pdf_name': pdf.pdf_name,
                    'page': 0,
                    'label': ex_label,
                    'data_gt': normalize_string( abs )
                })
        
        elif ex_label == "section" and key == "section":
            for sec in value:
                # sec might be a dict {"number","title"} or just a string
                title = sec['title'] if isinstance(sec, dict) else sec
                results.append({
                    'tool'     : tool,
                    'pdf_name' : pdf.pdf_name,
                    'page'     : 0,
                    'label'    : ex_label,
                    'data_gt'  : normalize_string ( title )
                })

        elif ex_label == "reference" and key == "bibliography": 
            for bib_entry in value :
                results.append({
                    'tool' : tool,
                    'pdf_name': pdf.pdf_name,
                    'page': 0,
                    'label': ex_label,
                    'data_gt': normalize_string( bib_entry['text'] )
                })
        
        elif ex_label == "pub_date" and key == "date":
            results.append({
                'tool' : tool,
                'pdf_name': pdf.pdf_name,
                'page': 0,
                'label': ex_label,
                'data_gt': normalize_string( value )
            })
        
        elif ex_label == key :
            if isinstance(value, list) :
                for elem in value :
                    results.append({
                    'tool' : tool,
                    'pdf_name': pdf.pdf_name,
                    'page': 0,
                    'label': ex_label,
                    'data_gt': normalize_string( elem )
                })
            else :
                results.append({
                'tool' : tool,
                'pdf_name': pdf.pdf_name,
                'page': 0,
                'label': ex_label,
                'data_gt': normalize_string( value )
            })
    return results
        