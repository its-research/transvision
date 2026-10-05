"""Stage admitted ClearML bytes then execute canonical fit-only calibration on CPU."""
import base64,hashlib,json,os,sys,subprocess,tarfile,shutil,time
from pathlib import Path,PurePosixPath
from urllib.parse import urlparse,urlunparse
from clearml import Task
from clearml.backend_api.session import Session
from clearml.storage.helper import StorageHelper
RUNNER_B64='IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJFeGVjdXRlIHVuY2hhbmdlZCBjYW5vbmljYWwgZml0IGFuZCBpbmRlcGVuZGVudCByZWNvbnN0cnVjdGlvbiBvbiBhZG1pdHRlZCBMaW51eCBDUFUuCgpUaGlzIHJ1bm5lciBpcyBub3QgYSBib290c3RyYXAgb3IgYW4gZXhwZXJpbWVudCBhY2NlcHRhbmNlIHJlY2VpcHQuIEl0cyBpbnB1dHMKbXVzdCBmaXJzdCBiZSBzdGFnZWQgZnJvbSBpbmRlcGVuZGVudGx5IGFkbWl0dGVkIENsZWFyTUwgYXJ0aWZhY3RzLgoiIiIKaW1wb3J0IGFyZ3BhcnNlLGhhc2hsaWIsanNvbixvcyxwbGF0Zm9ybSxzdWJwcm9jZXNzLHN5cwpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKZnJvbSBpbXBvcnRsaWIubWV0YWRhdGEgaW1wb3J0IHZlcnNpb24sZGlzdHJpYnV0aW9uClNPVVJDRV9NQU5JRkVTVF9TSEE9JzJkMDFmNWQwYmZjOTRjNDgzN2VlZGQ1NzRlZjZiMjE3NmY4MjExZGM0MjgzNTJiNDlkMjc1NzNhODUzM2M1MTcnCk1FVEFEQVRBX01BTklGRVNUX1NIQT0nOTA3ZTVhMDk4NmRhYTRjNDZhNzQ3ZDBmNzM0NGQ1YTVmM2VkZjBmN2RkMWVkNzE4ZmIzN2M0NzEwYWUyODQ5MicKRklUVEVSX1NIQT0nOTkxMWJhZjJiNjRlNzZjZTFhZjk4ZDhjYTRjMDY0ODAxMmYzOTI0ZmM4MTE5MmQ1OTQ5MWRhNDgwNTZkYjI0MScKVkVSSUZJRVJfU0hBPSc3MjE3N2UwNGYzZDczN2Q2OGFiNGVhNjE1ODRkMjc0M2YzOTU2OGYwYmY3YTk5NzdiMTdhN2JmM2E3ZDAwMWI3JwpSVU5USU1FX1RBU0s9Jzk2OWU1NmMwMzQ3MTRkZmFiMTBlMTY0NzFkNjhkN2UxJwpQQUNLQUdFX1JFQ09SRFM9eydudW1weSc6Jzg0Yzc2MzY2NzUzMzgzYTZlNjNmZmY3ZDkzYzU4ZmY5NjRkN2EyYzRhODk0YjczZGQxMmQ4ZjAxMWM4YWI0NWQnLCdzY2lweSc6JzRjNzhiMWI5ZmNlMWIxNTc3YTFkOGQ5YWRmMmU4ZTBiMTM5YTU3MDgwODU0MDllYzcxNjg0MDMwNDZmZWE0YjMnLCdjbGVhcm1sJzonYTY1OTI2NDU0ZmNjMWM5NWU1YTVkNjY5YThkNDg5NDg4MmRkNTRkMDY2MGFlMTU1MDRhMzBkNWM1ZGUxNjMyYyd9ClJBV19TT1VSQ0U9JzVkMjE4Njg1OTk1NTQ4OTQ4Y2NiMzdiOTJkZDMzN2RlJwpSQVdfQUNDRVBUT1I9J2ZlZmQ2NTFhMzNmNmRhYmY1NmVhMDYxMmVlYmM4MDY3M2FmNWU1ZDUwYjc0NjZkMGU5Mjg2NDRjNzU2OWY4NzgnCmRlZiBuZWVkKG9rLG1lc3NhZ2UpOgogaWYgbm90IG9rOnJhaXNlIFZhbHVlRXJyb3IobWVzc2FnZSkKZGVmIHNoYShwKToKIGg9aGFzaGxpYi5zaGEyNTYoKQogd2l0aCBQYXRoKHApLm9wZW4oJ3JiJykgYXMgZjoKICBmb3IgYiBpbiBpdGVyKGxhbWJkYTpmLnJlYWQoMTAyNCoqMiksYicnKTpoLnVwZGF0ZShiKQogcmV0dXJuIGguaGV4ZGlnZXN0KCkKZGVmIGludmVudG9yeShyb290LHJvd3MpOgogc2Vlbj1zZXQoKQogZm9yIHJvdyBpbiByb3dzOgogIG5hbWU9UGF0aChyb3dbJ3BhdGgnXSk7bmVlZChub3QgbmFtZS5pc19hYnNvbHV0ZSgpIGFuZCAnLi4nIG5vdCBpbiBuYW1lLnBhcnRzIGFuZCByb3dbJ3BhdGgnXSBub3QgaW4gc2VlbiwndW5zYWZlIGludmVudG9yeScpCiAgcD1yb290L25hbWU7bmVlZChub3QgcC5pc19zeW1saW5rKCkgYW5kIHAuaXNfZmlsZSgpIGFuZCBwLnN0YXQoKS5zdF9zaXplPT1yb3dbJ2J5dGVzJ10gYW5kIHNoYShwKT09cm93WydzaGEyNTYnXSwnaW52ZW50b3J5IGZpbGUgY2hhbmdlZCcpCiAgc2Vlbi5hZGQocm93WydwYXRoJ10pCiBuZWVkKG5vdCByb290LmlzX3N5bWxpbmsoKSBhbmQgbm90IGFueShwLmlzX3N5bWxpbmsoKSBmb3IgcCBpbiByb290LnJnbG9iKCcqJykpLCdzeW1saW5rZWQgaW52ZW50b3J5IHJvb3QnKQogYWN0dWFsPXtwLnJlbGF0aXZlX3RvKHJvb3QpLmFzX3Bvc2l4KCkgZm9yIHAgaW4gcm9vdC5yZ2xvYignKicpIGlmIHAuaXNfZmlsZSgpfQogbmVlZChhY3R1YWw9PXNlZW4sJ2V4dHJhIG9yIG1pc3NpbmcgaW52ZW50b3J5IGZpbGUnKQpkZWYgbWFpbigpOgogcGFyc2VyPWFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKGRlc2NyaXB0aW9uPV9fZG9jX18pCiBmb3IgbmFtZSBpbiAoJ3NvdXJjZS1yb290Jywnc291cmNlLW1hbmlmZXN0JywnbWV0YWRhdGEtcm9vdCcsJ21ldGFkYXRhLW1hbmlmZXN0Jywnc3VwZXJ2aXNpb24tcm9vdCcsJ2ZpdC1pbnB1dHMnLCdyYXctcmVhZGJhY2snLCdvdXRwdXQnKTpwYXJzZXIuYWRkX2FyZ3VtZW50KCctLScrbmFtZSx0eXBlPVBhdGgscmVxdWlyZWQ9VHJ1ZSkKIHBhcnNlci5hZGRfYXJndW1lbnQoJy0tcmF3LXJlYWRiYWNrLXNoYTI1NicscmVxdWlyZWQ9VHJ1ZSk7cGFyc2VyLmFkZF9hcmd1bWVudCgnLS1mb2xkJyx0eXBlPWludCxjaG9pY2VzPSgwLDEpLHJlcXVpcmVkPVRydWUpCiBhPXBhcnNlci5wYXJzZV9hcmdzKCk7bmVlZChub3QgYS5vdXRwdXQuZXhpc3RzKCksJ291dHB1dCBleGlzdHM7IHByZXNlcnZlIHByaW9yIGF0dGVtcHQnKQogbmVlZChwbGF0Zm9ybS5zeXN0ZW0oKT09J0xpbnV4JyBhbmQgcGxhdGZvcm0ubWFjaGluZSgpPT0neDg2XzY0JyBhbmQgc3lzLnZlcnNpb25faW5mb1s6Ml09PSgzLDEyKSwnQ1BVIGludGVycHJldGVyL3BsYXRmb3JtIGRpZmZlcnMnKQogbmVlZChwbGF0Zm9ybS5weXRob25fdmVyc2lvbigpPT0nMy4xMi4zJywnYWRtaXR0ZWQgTGludXggaW50ZXJwcmV0ZXIgdmVyc2lvbiBkaWZmZXJzJykKIG5lZWQob3MuZW52aXJvbi5nZXQoJ0NVREFfVklTSUJMRV9ERVZJQ0VTJyk9PScnLCdDUFUgcnVudGltZSBtdXN0IGhpZGUgQ1VEQScpCiBmb3Iga2V5IGluICgnT01QX05VTV9USFJFQURTJywnT1BFTkJMQVNfTlVNX1RIUkVBRFMnLCdNS0xfTlVNX1RIUkVBRFMnLCdOVU1FWFBSX05VTV9USFJFQURTJyk6bmVlZChvcy5lbnZpcm9uLmdldChrZXkpPT0nMScsJ0NQVSB0aHJlYWQgcG9saWN5IGRpZmZlcnMnKQogdmVyc2lvbnM9e2tleTp2ZXJzaW9uKGtleSkgZm9yIGtleSBpbiAoJ251bXB5Jywnc2NpcHknLCdjbGVhcm1sJyl9O25lZWQodmVyc2lvbnM9PXsnbnVtcHknOicxLjI2LjQnLCdzY2lweSc6JzEuMTQuMScsJ2NsZWFybWwnOicyLjEuNSd9LCdMaW51eDEyNiBwYWNrYWdlIHZlcnNpb25zIGRpZmZlcicpCiByZWNvcmRzPXt9CiBmb3Iga2V5LGV4cGVjdGVkIGluIFBBQ0tBR0VfUkVDT1JEUy5pdGVtcygpOgogIHJlY29yZD1kaXN0cmlidXRpb24oa2V5KS5yZWFkX3RleHQoJ1JFQ09SRCcpO25lZWQocmVjb3JkIGlzIG5vdCBOb25lLCdtaXNzaW5nIHBhY2thZ2UgaW5zdGFsbGF0aW9uIHJlY29yZCcpCiAgcmVjb3Jkc1trZXldPWhhc2hsaWIuc2hhMjU2KHJlY29yZC5lbmNvZGUoKSkuaGV4ZGlnZXN0KCk7bmVlZChyZWNvcmRzW2tleV09PWV4cGVjdGVkLCdwYWNrYWdlIGluc3RhbGxhdGlvbiByZWNvcmQgZGlmZmVycycpCiBuZWVkKHNoYShhLnNvdXJjZV9tYW5pZmVzdCk9PVNPVVJDRV9NQU5JRkVTVF9TSEEgYW5kIHNoYShhLm1ldGFkYXRhX21hbmlmZXN0KT09TUVUQURBVEFfTUFOSUZFU1RfU0hBLCdjbG91ZCBzb3VyY2UvbWV0YWRhdGEgbWFuaWZlc3QgZGlmZmVycycpCiBzPWpzb24ubG9hZHMoYS5zb3VyY2VfbWFuaWZlc3QucmVhZF9ieXRlcygpKTttPWpzb24ubG9hZHMoYS5tZXRhZGF0YV9tYW5pZmVzdC5yZWFkX2J5dGVzKCkpO2ludmVudG9yeShhLnNvdXJjZV9yb290LHNbJ2ludmVudG9yeSddKTtpbnZlbnRvcnkoYS5tZXRhZGF0YV9yb290LG1bJ2ludmVudG9yeSddKQogdG9vbD1hLnNvdXJjZV9yb290Lyd0b29scy9ldmVudF90cmFja192MngnO2ZpdHRlcj10b29sLydmaXRfc3BkX2Nhbm9uaWNhbF9vb2ZfY2FsaWJyYXRpb25fcnVudGltZV92M19jYW5kaWRhdGUucHknO3ZlcmlmaWVyPXRvb2wvJ3ZlcmlmeV9zcGRfY2Fub25pY2FsX2NhbGlicmF0aW9uX2V4YW1wbGVzX3J1bnRpbWVfdjNfY2FuZGlkYXRlLnB5JztuZWVkKHNoYShmaXR0ZXIpPT1GSVRURVJfU0hBIGFuZCBzaGEodmVyaWZpZXIpPT1WRVJJRklFUl9TSEEsJ2ZpdHRlci9vcmFjbGUgc291cmNlIGRpZmZlcnMnKQogbmVlZChub3QgYS5yYXdfcmVhZGJhY2suaXNfc3ltbGluaygpIGFuZCBzaGEoYS5yYXdfcmVhZGJhY2spPT1hLnJhd19yZWFkYmFja19zaGEyNTYsJ2ZpdCByYXcgYWRtaXNzaW9uIGJ5dGVzIGRpZmZlcicpCiByPWpzb24ubG9hZHMoYS5yYXdfcmVhZGJhY2sucmVhZF9ieXRlcygpKTtuZWVkKHIuZ2V0KCdraW5kJyk9PSdzcGRfY2Fub25pY2FsX29vZl9maXRfZmVhdHVyZV9yYXdfcG9zZV92Ml9jbG91ZF9pbmRlcGVuZGVudF9jb250ZW50X3JlYWRiYWNrJyBhbmQgci5nZXQoJ3N0YXR1cycpPT0nYWxsX2Nsb3VkX2J5dGVzX2ZyYW1lc19hcnJheXNfcmF3X3Bvc2VzX3ZlcmlmaWVkJyBhbmQgci5nZXQoJ2ZvbGRfaWQnKT09YS5mb2xkIGFuZCByLmdldCgnc291cmNlX3Rhc2tfaWQnKT09UkFXX1NPVVJDRSBhbmQgci5nZXQoJ2FjY2VwdG9yX3NoYTI1NicpPT1SQVdfQUNDRVBUT1IgYW5kIHIuZ2V0KCdoZWxkX291dF9zZWxlY3Rpb25fc2NvcmluZ19lbGlnaWJsZScpIGlzIEZhbHNlLCdjb21wbGV0ZSBmaXQgcmF3IGFjY2VwdGFuY2UgbWlzc2luZycpCiBmcm9tIGNsZWFybWwgaW1wb3J0IFRhc2sKIHByb2JlPVRhc2suZ2V0X3Rhc2sodGFza19pZD1SVU5USU1FX1RBU0spO25lZWQocHJvYmUuc3RhdHVzPT0nY29tcGxldGVkJyBhbmQgcHJvYmUuYXJ0aWZhY3RzWydydW50aW1lLXByb2JlJ10uaGFzaD09JzJmYzg4NWM3NmY0NmRlNGJhNzc1NmFlYzUzN2RhYWU5NjBkYzJjYTA0MzgwMmFjMjM3ZTFiY2U3NzM5NWI2MmEnLCdhZG1pdHRlZCBMaW51eCBDUFUgcHJvYmUgcmVnaXN0cmF0aW9uIGRpZmZlcnMnKQogIyBQcmVzZXJ2ZSBvcmlnaW5hbCBhdWRpdCBST09UIHVuY2hhbmdlZC4gVGhlIGlzb2xhdGVkIGJvb3RzdHJhcCBtdXN0IHN0YWdlCiAjIHRoZSBleGFjdCBmaXZlIGFkbWl0dGVkIG1hbmlmZXN0cyB0aGVyZSwgd2l0aG91dCBwcmVkaWN0aW9uL0dUIHBheWxvYWRzLgogYWJzb2x1dGU9UGF0aCgnL1ZvbHVtZXMvRGF0YS90ZXN0L3JlY292ZXItYmVmb3JlLWZ1c2UvYXJ0aWZhY3RzL3NwZC1jYW5vbmljYWwtb29mLWhlbGRvdXQtaW5mZXJlbmNlLWlucHV0cy0yMDI2MDkzMCcpCiBmb3IgZm9sZCBpbiByYW5nZSg1KTpuZWVkKHNoYShhYnNvbHV0ZS9mJ2ZvbGQte2ZvbGR9L2lucHV0LW1hbmlmZXN0Lmpzb24nKT09c2hhKGEubWV0YWRhdGFfcm9vdC9mJ2hlbGRvdXQtbWFuaWZlc3RzL2ZvbGQte2ZvbGR9L2lucHV0LW1hbmlmZXN0Lmpzb24nKSwnb3JpZ2luYWwgYXVkaXQgbWFuaWZlc3QgbW91bnQgZGlmZmVycycpCiBwYWNrYWdlPWEuc3VwZXJ2aXNpb25fcm9vdC8ncGFja2FnZSc7Y29udmVydGVkPWEuc3VwZXJ2aXNpb25fcm9vdC8nY29udmVydGVkJztieXRlX2ZyZWV6ZT1hLm1ldGFkYXRhX3Jvb3QvZidieXRlLWZyZWV6ZXMvZm9sZC17YS5mb2xkfS5qc29uJztwb3Nlcz1hLm1ldGFkYXRhX3Jvb3QvJ3Jhdy1wb3Nlcyc7aGVsZD1hLm1ldGFkYXRhX3Jvb3QvZidoZWxkb3V0LW1hbmlmZXN0cy9mb2xkLXthLmZvbGR9JwogY29tbW9uPVsnLS1wYWNrYWdlJyxzdHIocGFja2FnZSksJy0tY29udmVydGVkJyxzdHIoY29udmVydGVkKSwnLS1maXQtaW5wdXRzJyxzdHIoYS5maXRfaW5wdXRzKSwnLS1oZWxkb3V0LWlucHV0cycsc3RyKGhlbGQpLCctLXJhdy1yZWFkYmFjaycsc3RyKGEucmF3X3JlYWRiYWNrKSwnLS1yYXctcmVhZGJhY2stc2hhMjU2JyxhLnJhd19yZWFkYmFja19zaGEyNTYsJy0tYnl0ZS1mcmVlemUnLHN0cihieXRlX2ZyZWV6ZSksJy0tcmF3LXBvc2VzJyxzdHIocG9zZXMpXQogIyBDaGlsZCBpbXBvcnQgcGF0aCBjb250YWlucyB0aGUgYWRtaXR0ZWQgc291cmNlIG9ubHk7IG5vIGFtYmllbnQgcmVwb3NpdG9yeS4KIGVudj1kaWN0KG9zLmVudmlyb24pO2Vudi51cGRhdGUoUFlUSE9OTk9VU0VSU0lURT0nMScsUFlUSE9ORE9OVFdSSVRFQllURUNPREU9JzEnKTtlbnYucG9wKCdQWVRIT05QQVRIJyxOb25lKQogcnVubmVyPSJpbXBvcnQgcnVucHksc3lzO3N5cy5wYXRoWzowXT1bc3lzLmFyZ3YucG9wKDEpLHN5cy5hcmd2LnBvcCgxKV07cnVucHkucnVuX3BhdGgoc3lzLmFyZ3YucG9wKDEpLHJ1bl9uYW1lPSdfX21haW5fXycpIgogZGVmIGV4ZWN1dGUoc2NyaXB0LGFyZ3MpOnN1YnByb2Nlc3MucnVuKFtzeXMuZXhlY3V0YWJsZSwnLUknLCctYycscnVubmVyLHN0cihhLnNvdXJjZV9yb290KSxzdHIodG9vbCksc3RyKHNjcmlwdCksKmFyZ3NdLGVudj1lbnYsY2hlY2s9VHJ1ZSkKIGEub3V0cHV0Lm1rZGlyKHBhcmVudHM9VHJ1ZSkKIHJ1bnRpbWU9ZGljdChraW5kPSdjYW5vbmljYWxfY2FsaWJyYXRpb25fbGludXgxMjZfYWN0dWFsX2NwdV9leGVjdXRpb25faWRlbnRpdHknLGludGVycHJldGVyPXN5cy52ZXJzaW9uLGV4ZWN1dGFibGU9c3lzLmV4ZWN1dGFibGUscGxhdGZvcm09cGxhdGZvcm0ucGxhdGZvcm0oKSx2ZXJzaW9ucz12ZXJzaW9ucyxwYWNrYWdlX3JlY29yZF9zaGEyNTY9cmVjb3JkcyxjdWRhX3Zpc2libGVfZGV2aWNlcz0nJyxydW50aW1lX3Byb2JlX3Rhc2tfaWQ9UlVOVElNRV9UQVNLLHJ1bm5lcl9zaGEyNTY9c2hhKFBhdGgoX19maWxlX18pKSxzb3VyY2VfbWFuaWZlc3Rfc2hhMjU2PXNoYShhLnNvdXJjZV9tYW5pZmVzdCksbWV0YWRhdGFfbWFuaWZlc3Rfc2hhMjU2PXNoYShhLm1ldGFkYXRhX21hbmlmZXN0KSxyYXdfZml0X3JlYWRiYWNrX3NoYTI1Nj1zaGEoYS5yYXdfcmVhZGJhY2spLGZvbGRfaWQ9YS5mb2xkLGZvcm1hbF92Ml9yZWFkeT1GYWxzZSxwYXBlcl9lbGlnaWJsZT1GYWxzZSkKIChhLm91dHB1dC8ncnVudGltZS1pZGVudGl0eS5qc29uJykud3JpdGVfdGV4dChqc29uLmR1bXBzKHJ1bnRpbWUsaW5kZW50PTIpKydcbicpCiBwcmludCgnQ0FMSUJSQVRJT05fUEhBU0UgZml0X2NvbGxlY3Rpb25fYW5kX29wdGltaXphdGlvbiBvdmVyYWxsX2V0YT11bmtub3duJyxmbHVzaD1UcnVlKQogY2FuZGlkYXRlPWEub3V0cHV0LydjYW5kaWRhdGUnO2V4ZWN1dGUoZml0dGVyLGNvbW1vbitbJy0tb3V0cHV0JyxzdHIoY2FuZGlkYXRlKV0pCiBjPWNhbmRpZGF0ZS8nY2FsaWJyYXRpb24uanNvbic7cmVjZWlwdD1hLm91dHB1dC8naW5kZXBlbmRlbnQtZXhhbXBsZXMtYW5kLXBhcmFtZXRlcnMuanNvbicKIHByaW50KCdDQUxJQlJBVElPTl9QSEFTRSBpbmRlcGVuZGVudF9mdWxsX2V4YW1wbGVfYW5kX3BhcmFtZXRlcl9yZWFkYmFjayBvdmVyYWxsX2V0YT11bmtub3duJyxmbHVzaD1UcnVlKQogZXhlY3V0ZSh2ZXJpZmllcixjb21tb24rWyctLWNhbGlicmF0aW9uJyxzdHIoYyksJy0tY2FsaWJyYXRpb24tc2hhMjU2JyxzaGEoYyksJy0tcmVjZWlwdCcsc3RyKHJlY2VpcHQpXSkKIHByb29mPWpzb24ubG9hZHMocmVjZWlwdC5yZWFkX2J5dGVzKCkpO25lZWQocHJvb2ZbJ2ZvbGRfaWQnXT09YS5mb2xkIGFuZCBwcm9vZlsncmF3X0dUX2V4YW1wbGVzX2luZGVwZW5kZW50bHlfcmVjb25zdHJ1Y3RlZCddIGlzIFRydWUgYW5kIHByb29mWydpbmRlcGVuZGVudF9wYXJhbWV0ZXJzX3JlY29tcHV0ZWQnXSBpcyBUcnVlIGFuZCBwcm9vZlsnaGVsZF9vdXRfR1RfdXNlZF9mb3JfZml0dGluZyddIGlzIEZhbHNlLCdpbmRlcGVuZGVudCBjYWxpYnJhdGlvbiByZXN1bHQgaW5jb21wbGV0ZScpCiBwcmludCgnQ0FMSUJSQVRJT05fQ1BVX0NBTkRJREFURV9JTkRFUEVOREVOVF9SRUFEQkFDS19DT01QTEVURSAnK3NoYShyZWNlaXB0KSxmbHVzaD1UcnVlKQppZiBfX25hbWVfXz09J19fbWFpbl9fJzptYWluKCkK'
RUNNER_SHA='f8b9fb992d0830a1a890c70f2a05ea80c7044a93ac2a4a13d41a76b8b1424455'

SOURCE_TASK='641448cbbb83421ab6c63b8537c3bf99'
SOURCE_SHA='2d01f5d0bfc94c4837eedd574ef6b2176f8211dc428352b49d27573a8533c517'
METADATA_TASK='c5ca5855aed84f6e83ceac022578f5ed'
METADATA_SHA='907e5a0986daa4c46a747d0f7344d5a5f3edf0f7dd1ed718fb37c4710ae28492'
SUPERVISION={0:('9459a23705a24c1c89a2fa7613268815','1866e2f0372da5dc4fe9ebec3975fb7231b8dfbf05d286aff0cffae930d9ce13'),1:('e554bf277e58439382e30900889af509','1cbd02a1a8b613d15230e756353c2d62964bfe48a685e17472a64d833deff260')}
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def fetch(owner,name,expected,path):
 need(owner.status=='completed','artifact owner not completed');item=owner.artifacts[name];u=urlparse(item.url);active=urlparse(Session.get_files_server_host())
 for url in (u,active):need(url.scheme=='http' and url.netloc in {'10.100.35.118:8081','10.100.34.118:8081'} and not(url.username or url.password or url.query or url.fragment),'unadmitted file-service route')
 need(item.hash==expected,'registered artifact SHA differs');routed=urlunparse(u._replace(netloc=active.netloc));h=hashlib.sha256();received=0;start=time.monotonic()
 with path.open('xb') as out:
  for block in StorageHelper.get(routed).download_as_stream(routed,chunk_size=1024**2):
   out.write(block);h.update(block);received+=len(block);need(received<=item.size,'oversized artifact')
   if received//(256*1024**2)>(received-len(block))//(256*1024**2):print('CALIBRATION_DOWNLOAD_PROGRESS '+json.dumps({'artifact':name,'bytes':received,'total':item.size,'eta_seconds':(item.size-received)*(time.monotonic()-start)/received,'scope':'this artifact only','overall_eta':'unknown'}),flush=True)
 need(received==item.size and h.hexdigest()==expected,'download byte count/SHA differs');return path
def extract_exact(archive,root,inventory):
 need(not root.exists(),'extraction destination exists');root.mkdir(parents=True);expected={r['path']:r for r in inventory};need(len(expected)==len(inventory),'duplicate archive inventory');seen=set()
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   name=PurePosixPath(m.name);need(m.isfile() and not name.is_absolute() and '..' not in name.parts and m.name in expected and m.name not in seen,'unsafe/extra archive member');row=expected[m.name];need(m.size==row['bytes'],'member size differs');path=root/m.name;path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
   with t.extractfile(m) as source,path.open('xb') as out:
    for b in iter(lambda:source.read(1024**2),b''):h.update(b);out.write(b)
   need(path.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'],'member bytes differ');seen.add(m.name)
 need(seen==set(expected),'incomplete archive')
def extract_cache(archive, destination, manifest, manifest_bytes):
    expected = {'raw-cache-manifest.json': {'bytes': len(manifest_bytes), 'sha256': hashlib.sha256(manifest_bytes).hexdigest()}, 'launch-receipt.json': None, 'resolved-cache-config.py': {'sha256': manifest['resolved_config_sha256']}}
    for frame in manifest['frames']:
        for key in ('arrays', 'metadata'):
            row = frame[key]
            relative = PurePosixPath(row['path'])
            need(not relative.is_absolute() and '..' not in relative.parts and (relative.as_posix() not in expected), 'unsafe or duplicate cache member')
            expected[relative.as_posix()] = row
    need(not destination.exists(), 'cache extraction is create-once')
    destination.mkdir()
    seen = set()
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            relative = PurePosixPath(member.name)
            need(member.isfile() and (not relative.is_absolute()) and ('..' not in relative.parts) and (relative.parts[0] == 'cache'), 'non-file or unsafe archive member')
            name = PurePosixPath(*relative.parts[1:]).as_posix()
            need(name in expected and name not in seen, 'extra or duplicate archive file')
            row = expected[name]
            if row and 'bytes' in row:
                need(member.size == row['bytes'], 'archive member size differs')
            else:
                need(member.size < 8 * 1024 * 1024, 'oversized cache header')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            count = 0
            with tf.extractfile(member) as source, target.open('xb') as out:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    h.update(block)
                    count += len(block)
                    out.write(block)
            need(count == member.size, 'truncated archive member')
            if row and 'sha256' in row:
                need(h.hexdigest() == row['sha256'], 'archive member bytes differ')
            seen.add(name)
    need(seen == set(expected), 'archive inventory incomplete')
    return len(seen)

def main():
 task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='SPD canonical fit-only calibration Linux126 CPU',reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
 p={k.split('/',1)[-1]:v for k,v in task.get_parameters().items() if k.startswith('General/')};fold=p['fold_id'];need(type(fold) is int and fold in SUPERVISION,'unadmitted fold')
 text=p['raw_fit_readback_text'];need(hashlib.sha256(text.encode()).hexdigest()==p['raw_fit_readback_sha256'],'fit admission text differs');raw=json.loads(text)
 need(raw.get('status')=='all_cloud_bytes_frames_arrays_raw_poses_verified' and raw.get('fold_id')==fold and raw.get('source_task_id')=='5d218685995548948ccb37b92dd337de' and raw.get('acceptor_sha256')=='fefd651a33f6dabf56ea0612eebc80673af5e5d50b7466d0e928644c7569f878' and raw.get('held_out_selection_scoring_eligible') is False,'complete fit admission absent')
 need(p['runner_sha256']==RUNNER_SHA,'CPU runner differs');need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU CUDA policy differs')
 root=Path('/eventtrack-canonical-calibration-linux126');root.mkdir();downloads=root/'downloads';downloads.mkdir()
 def bundle(ownerid,manifestname,archivesname,manifestsha,filename,target):
  owner=Task.get_task(task_id=ownerid);mp=fetch(owner,manifestname,manifestsha,downloads/(filename+'.json'));m=json.loads(mp.read_bytes());ar=fetch(owner,archivesname,m['archive']['sha256'],downloads/(filename+'.tar.gz'));need(ar.stat().st_size==m['archive']['bytes'],'archive registered size differs');extract_exact(ar,target,m['inventory']);return mp,m
 source_manifest,s=bundle(SOURCE_TASK,'source-manifest','source-archive',SOURCE_SHA,'source',root/'source')
 metadata_manifest,m=bundle(METADATA_TASK,'source-manifest','source-archive',METADATA_SHA,'metadata',root/'metadata')
 supid,supsha=SUPERVISION[fold];supervision_manifest,sup=bundle(supid,'supervision-manifest','fit-only-supervision',supsha,'supervision',root/'supervision')
 need(sup['fold_id']==fold and sup['held_out_gt_read'] is False,'supervision role differs')
 absolute=Path('/Volumes/Data/test/recover-before-fuse/artifacts/spd-canonical-oof-heldout-inference-inputs-20260930');need(not absolute.exists(),'absolute audit mount exists');absolute.mkdir(parents=True)
 for fid in range(5):
  target=absolute/f'fold-{fid}';target.mkdir();shutil.copyfile(root/f'metadata/heldout-manifests/fold-{fid}/input-manifest.json',target/'input-manifest.json')
 sys.path[:0]=[str(root/'metadata/download-helpers'),str(root/'source/tools/event_track_v2x'),str(root/'source')]
 from download_spd_oof_fit_feature_cloud_inputs import download_fit_inputs
 cloud=(root/'metadata/cloud-input-evidence.json').read_text();overlay=(root/'metadata/fit-overlay-admission.json').read_text()
 fit,fit_proof=download_fit_inputs(cloud,overlay,root/'supervision/package/package-manifest.json',fold,root/'fit-input-build')
 need(fit_proof['status']=='passed' and fit_proof['all_reconstructed_payload_bytes_verified'] is True,'fit input reconstruction failed')
 need(sha(fit/'input-manifest.json')==sha(root/'supervision/fit-inputs/input-manifest.json'),'full reconstructed fit complement differs from supervision')
 rawfolder=root/'raw';rawfolder.mkdir();raw_receipt=rawfolder/'acceptance-receipt.json';raw_receipt.write_text(text);owner=Task.get_task(task_id=raw['task_id']);need(owner.status=='completed' and set(owner.artifacts)==set(raw['artifacts']),'fit task artifact scope differs')
 for name,row in raw['artifacts'].items():
  rel=PurePosixPath(row['path']);need(not rel.is_absolute() and '..' not in rel.parts and len(rel.parts)==1,'unsafe raw artifact filename');target=rawfolder/rel;fetch(owner,name,row['sha256'],target);need(target.stat().st_size==row['bytes'],'fit artifact size differs')
 for side in ('vehicle-side','infrastructure-side'):
  for shard in (0,1):
   prefix=f'{side}-shard-{shard}';mp=rawfolder/raw['artifacts'][prefix+'-raw-manifest']['path'];manifest=json.loads(mp.read_bytes());archive=rawfolder/raw['artifacts'][prefix+'-raw-cache']['path'];extract_cache(archive,rawfolder/(prefix+'-cache'),manifest,mp.read_bytes())
 runner=root/'runner.py';runner.write_bytes(base64.b64decode(RUNNER_B64));need(sha(runner)==RUNNER_SHA,'embedded runner bytes differ')
 args=[sys.executable,str(runner),'--fold',str(fold),'--source-root',str(root/'source'),'--source-manifest',str(source_manifest),'--metadata-root',str(root/'metadata'),'--metadata-manifest',str(metadata_manifest),'--supervision-root',str(root/'supervision'),'--fit-inputs',str(fit),'--raw-readback',str(raw_receipt),'--raw-readback-sha256',p['raw_fit_readback_sha256'],'--output',str(root/'results')]
 subprocess.run(args,check=True)
 results=root/'results';artifacts={};paths=list((results/'candidate').glob('*'))+[results/'runtime-identity.json',results/'independent-examples-and-parameters.json']
 for path in paths:
  name=path.stem;need(name not in artifacts and path.is_file(),'duplicate/non-file output');need(task.upload_artifact(name,artifact_object=path,wait_on_upload=True),'calibration publication failed');artifacts[name]=dict(sha256=sha(path),bytes=path.stat().st_size,path=path.name)
 task.reload()
 for name,row in artifacts.items():need(task.artifacts[name].hash==row['sha256'] and task.artifacts[name].size==row['bytes'],'registered calibration output differs')
 summary=dict(kind='canonical_calibration_linux126_cpu_fitted_and_job_local_reconstructed_summary',fold_id=fold,raw_fit_task_id=raw['task_id'],raw_fit_readback_sha256=p['raw_fit_readback_sha256'],runner_sha256=RUNNER_SHA,source_task_id=SOURCE_TASK,metadata_task_id=METADATA_TASK,supervision_task_id=supid,fit_materialization=fit_proof,command=args,artifacts=artifacts,independent_cloud_readback_verified=False,formal_v2_ready=False,paper_eligible=False)
 need(task.upload_artifact('execution-summary',artifact_object=summary,wait_on_upload=True),'summary publication failed')
 print('CALIBRATION_CPU_JOB_LOCAL_FINISHED cloud_acceptance=not_yet_verified',flush=True)
if __name__=='__main__':main()
