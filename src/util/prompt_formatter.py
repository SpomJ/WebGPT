ROLE_TOKEN = {
    'system': '@[SYS]',
    'user': '@[USR]',
    'web': '@[WEB]',
    'bot': '@[BOT]'
}
S_END = '@[END]'

def fmt(msgs: list):
    O = ''
    for msg in msgs:
      O += ROLE_TOKEN[msg['role']]
      O += msg['content']
      O += S_END
    return O
