ROLE_TOKEN = {
    'system': '@[SYS]',
    'user': '@[USR]',
    'web': '@[WEB]',
    'bot': '@[BOT]'
}
S_END = '@[END]'

def bulk_fmt(dataset: 'Dataset dict in form of {id: [1,2,...], messages: [[{...}, {...}], [...]]}'):
    O = []
    for chain in dataset['messages']:
        c = ''
        for msg in chain:
            c += ROLE_TOKEN[msg['role']]
            c += msg['content']
            c += S_END
        O.append(c)
    return O


def fmt_user_input(s):
    return ROLE_TOKEN['user'] + s + S_END
   
