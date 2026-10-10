/** Small RFC 4180 style CSV reader for the repository's numeric result files. */
export function parseCsv(text) {
  const source = String(text ?? '').replace(/^\uFEFF/, '')
  const records = []
  let row = []
  let field = ''
  let quoted = false

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index]
    if (quoted) {
      if (char === '"' && source[index + 1] === '"') {
        field += '"'
        index += 1
      } else if (char === '"') {
        quoted = false
      } else {
        field += char
      }
    } else if (char === '"' && field.length === 0) {
      quoted = true
    } else if (char === ',') {
      row.push(field.trim())
      field = ''
    } else if (char === '\n' || char === '\r') {
      if (char === '\r' && source[index + 1] === '\n') index += 1
      row.push(field.trim())
      if (row.some((value) => value !== '')) records.push(row)
      row = []
      field = ''
    } else {
      field += char
    }
  }
  if (quoted) throw new Error('CSV contains an unterminated quoted field')
  row.push(field.trim())
  if (row.some((value) => value !== '')) records.push(row)
  if (records.length === 0) return []

  const headers = records[0].map((header) => header.trim())
  if (headers.some((header) => !header) || new Set(headers).size !== headers.length) {
    throw new Error('CSV has a blank or duplicate column name')
  }
  return records.slice(1).map((values, rowIndex) => {
    if (values.length !== headers.length) {
      throw new Error(`CSV row ${rowIndex + 2} has ${values.length} values; expected ${headers.length}`)
    }
    return Object.fromEntries(headers.map((header, index) => {
      const raw = values[index]
      if (raw === '') return [header, '']
      const number = Number(raw)
      return [header, Number.isFinite(number) ? number : raw]
    }))
  })
}

export function rollingMean(rows, key, windowSize) {
  return rows.map((row, index) => {
    const start = Math.max(0, index - windowSize + 1)
    const slice = rows.slice(start, index + 1)
    const mean = slice.reduce((sum, item) => sum + Number(item[key] || 0), 0) / slice.length
    return { ...row, [`${key}_roll${windowSize}`]: mean }
  })
}
