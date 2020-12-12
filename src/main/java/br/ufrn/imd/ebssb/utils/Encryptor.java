package br.ufrn.imd.ebssb.utils;

import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class Encryptor {

	public static String encryptSh1(String id){
		// Encrypting
		MessageDigest algorithm;
		byte messageDigest[] = {};
		try {
			algorithm = MessageDigest.getInstance("SHA-1");
			messageDigest = algorithm.digest(id.getBytes("UTF-8"));
		} 
		catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		catch(NoSuchAlgorithmException f) {
			f.printStackTrace();
		}
		// Converting from byte to hexadecimal
		StringBuilder hexString = new StringBuilder();
		for (byte b : messageDigest) {
			hexString.append(String.format("%02X", 0xFF & b));
		}
		// Converting from hexadecimal to string
		String newId = hexString.toString();
		return newId;
	}
}
